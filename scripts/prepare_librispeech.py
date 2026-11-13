#!/usr/bin/env python3
"""Prepare LibriSpeech raw audio and manifests, without starting training.

Default: train-clean-100 + four dev/test splits (roughly 7.7 GB compressed).
Use --scope 960h for all three training splits + dev/test (roughly 61.4 GB).
Requires Python 3.12+ and curl; allow additional space for extraction.
Dataset: https://www.openslr.org/12
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import re
import time

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://www.openslr.org/resources/12/'
MIRRORS = {
    'primary': BASE_URL,
    'elda': 'https://openslr.elda.org/resources/12/',
    'trmal': 'https://openslr.trmal.net/resources/12/',
}
SPLITS = {
    'train-clean-100': '2a93770f6d5c6c964bc36631d331a522',
    'train-clean-360': 'c0e676e450a7ff2f54aeade5171606fa',
    'train-other-500': 'd1a0fd59409feb2c614ce4d30c387708',
    'dev-clean': '42e2234ba48799c1f50f24a7926300a1',
    'dev-other': 'c8d0bcc9cca99d4f8b62fcc847357931',
    'test-clean': '32fa31d27d2e1cad72775fee3f4849a9',
    'test-other': 'fb5a50374b501bb3bac4815ee91d3135',
}
DEFAULT_SPLITS = tuple(s for s in SPLITS if s not in ('train-clean-360', 'train-other-500'))
TRAIN_SPLITS = ('train-clean-100', 'train-clean-360', 'train-other-500')
EXPECTED_UTTERANCES = {
    'train-clean-100': 28539, 'train-clean-360': 104014,
    'train-other-500': 148688, 'dev-clean': 2703, 'dev-other': 2864,
    'test-clean': 2620, 'test-other': 2939,
}


def md5(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, destination, resume=False):
    command = ['curl','--fail','--location','--silent','--show-error','--proto','=https','--proto-redir','=https','--retry','3','--connect-timeout','30','--speed-limit','1024','--speed-time','90']
    if resume:
        command += ['--continue-at','-']
    subprocess.run(command + ['--output',str(destination),'--',url],check=True)


def fetch_ranged(url, destination, connections):
    """Use validated HTTPS byte ranges; keep completed segments for resumption."""
    headers = subprocess.check_output(['curl','--fail','--location','--silent','--show-error','--head','--proto','=https','--proto-redir','=https','--',url],text=True)
    lengths = re.findall(r'^content-length:\s*(\d+)',headers,re.I|re.M)
    if not lengths:
        fetch(url,destination,resume=True)
        return
    total = int(lengths[-1])
    chunk = 32*1024*1024
    ranges = [(start,min(start+chunk,total)-1) for start in range(0,total,chunk)]
    pieces = destination.with_name(destination.name+'.segments')
    pieces.mkdir(exist_ok=True)
    if destination.exists():
        with destination.open('rb') as old:
            for start,end in ranges:
                if end >= destination.stat().st_size:
                    break
                part = pieces/f'{start:012d}.bin'
                if not part.exists():
                    old.seek(start)
                    part.write_bytes(old.read(end-start+1))
    def download(bounds):
        start,end = bounds
        part = pieces/f'{start:012d}.bin'
        if part.exists() and part.stat().st_size == end-start+1:
            return
        response_headers = part.with_suffix('.headers')
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--proto','=https','--proto-redir','=https','--retry','3','--connect-timeout','30','--max-time','240','--range',f'{start}-{end}','--dump-header',str(response_headers),'--output',str(part),'--',url],check=True)
        content_ranges = re.findall(r'^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',response_headers.read_text(),re.I|re.M)
        if not content_ranges or tuple(map(int,content_ranges[-1])) != (start,end,total) or part.stat().st_size != end-start+1:
            part.unlink(missing_ok=True)
            raise ValueError('Server did not return the requested byte range')
    completed = 0
    last_update = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1,min(connections,16))) as pool:
        jobs = {pool.submit(download,bounds):bounds for bounds in ranges}
        for job in as_completed(jobs):
            job.result()
            start,end = jobs[job]
            completed += end-start+1
            if time.monotonic()-last_update >= 30 or completed == total:
                print(f'PROGRESS {destination.name}: {completed/1e9:.2f}/{total/1e9:.2f} GB ({completed/total:.1%})',flush=True)
                last_update = time.monotonic()
    print(f'MERGE {destination.name}',flush=True)
    merged = destination.with_name(destination.name+'.merged')
    with merged.open('wb') as out:
        for start,_ in ranges:
            with (pieces/f'{start:012d}.bin').open('rb') as source:
                shutil.copyfileobj(source,out,1024*1024)
    merged.replace(destination)
    shutil.rmtree(pieces)


def prepare_split(split, raw, official_checksums, connections, base_url=BASE_URL):
    archives = raw / 'archives'
    archives.mkdir(parents=True,exist_ok=True)
    archive = archives / (split+'.tar.gz')
    expected = SPLITS[split]
    if official_checksums.get(archive.name) != expected:
        raise ValueError(f'{split}: upstream checksum differs from the reviewed checksum')
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix+'.part')
        print(f'DOWNLOAD {split}',flush=True)
        fetch_ranged(base_url+archive.name,partial,connections)
        if md5(partial) != expected:
            raise ValueError(f'{split}: download checksum mismatch; partial file retained for inspection')
        partial.replace(archive)
    elif md5(archive) != expected:
        raise ValueError(f'{split}: existing archive checksum mismatch')
    marker = raw / ('verified-'+split+'.json')
    if not marker.exists():
        print(f'EXTRACT {split}',flush=True)
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                target = (raw/member.name).resolve()
                if not target.is_relative_to(raw.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsafe archive member: {member.name}')
                tar.extract(member,path=raw,filter='data')
        marker.write_text(json.dumps({'split':split,'archive_md5':expected,'source_url':base_url+archive.name},indent=2)+'\n')
    print(f'READY {split}',flush=True)
    provenance = json.loads(marker.read_text())
    return {'split':split,'archive_md5':expected,'archive_bytes':archive.stat().st_size,'archive_url':provenance['source_url']}


def flac_info(path):
    """Read sample metadata from the lossless FLAC STREAMINFO block."""
    with path.open('rb') as f:
        if f.read(4) != b'fLaC':
            raise ValueError(f'Not FLAC: {path.name}')
        while True:
            header = f.read(4)
            if len(header) != 4:
                raise ValueError(f'Missing STREAMINFO: {path.name}')
            block_type = header[0] & 0x7f
            length = int.from_bytes(header[1:4],'big')
            if block_type == 0:
                block = f.read(length)
                if length != 34 or len(block) != 34:
                    raise ValueError(f'Invalid STREAMINFO: {path.name}')
                packed = int.from_bytes(block[10:18],'big')
                rate = packed >> 44
                channels = ((packed >> 41) & 7) + 1
                samples = packed & ((1 << 36)-1)
                bits = ((packed >> 36) & 31) + 1
                if not rate or not samples:
                    raise ValueError(f'Empty audio metadata: {path.name}')
                return rate,channels,samples,bits
            f.seek(length,1)
            if header[0] & 0x80:
                raise ValueError(f'Missing STREAMINFO: {path.name}')


def inventory(raw, manifests, archive_records, splits=DEFAULT_SPLITS):
    manifests.mkdir(parents=True,exist_ok=True)
    reports = []
    speakers = {}
    all_ids = set()
    for split in splits:
        print(f'INVENTORY {split}',flush=True)
        folder = raw/'LibriSpeech'/split
        transcripts = {}
        for source in sorted(folder.rglob('*.trans.txt')):
            for line in source.read_text().splitlines():
                uid,text = line.split(' ',1)
                if uid in transcripts or not text.strip():
                    raise ValueError(f'Duplicate or empty transcript: {uid}')
                transcripts[uid] = text
        audios = sorted(folder.rglob('*.flac'))
        audio_ids = {p.stem for p in audios}
        if len(audios) != EXPECTED_UTTERANCES[split] or len(audio_ids) != len(audios):
            raise ValueError(f'{split}: unexpected or duplicate audio count ({len(audios)})')
        if audio_ids != set(transcripts):
            raise ValueError(f'{split}: audio/transcript IDs do not match')
        if all_ids & audio_ids:
            raise ValueError(f'{split}: duplicate utterance IDs across splits')
        all_ids |= audio_ids
        duration, word_count, max_duration = 0.0,0,0.0
        split_speakers = set()
        manifest = manifests/(split+'.jsonl')
        temporary_manifest = manifest.with_suffix('.jsonl.tmp')
        with temporary_manifest.open('w') as out:
            for path in audios:
                rate,channels,samples,bits = flac_info(path)
                if rate != 16000 or channels != 1 or bits != 16:
                    raise ValueError(f'{path.name}: unexpected audio format')
                uid = path.stem
                speaker,chapter,_ = uid.split('-')
                seconds = samples/rate
                row = {'dataset':'LibriSpeech','split':split,'utterance_id':uid,'speaker_id':speaker,'chapter_id':chapter,'audio':path.relative_to(ROOT).as_posix(),'text':transcripts[uid],'sample_rate':rate,'channels':channels,'num_samples':samples,'duration_seconds':seconds}
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
                duration += seconds
                max_duration = max(max_duration,seconds)
                word_count += len(transcripts[uid].split())
                split_speakers.add(speaker)
        temporary_manifest.replace(manifest)
        speakers[split] = split_speakers
        reports.append({'split':split,'utterances':len(audios),'speakers':len(split_speakers),'duration_hours':round(duration/3600,6),'words':word_count,'max_utterance_seconds':round(max_duration,3),'sample_rate':16000,'channels':1,'audio_transcript_pairs_verified':True})
    intersections = {}
    for i,a in enumerate(splits):
        for b in list(splits)[i+1:]:
            overlap = speakers[a] & speakers[b]
            intersections[a+' / '+b] = len(overlap)
            if overlap:
                raise ValueError(f'Speaker overlap: {a} / {b}')
    report = {'dataset':'LibriSpeech','source':'https://www.openslr.org/12','license':'CC BY 4.0','prepared_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'stage':'raw audio and utterance manifests only; this preparation run does not create features, tokenizer, forced alignments or trained models','archive_verification':'MD5 compared against official md5sum.txt and reviewed checksums','audio_verification':'All FLAC STREAMINFO headers inspected; archive checksum protects full file payload. Not every waveform was decoded.','archives':sorted(archive_records,key=lambda r:r['split']),'splits':reports,'speaker_overlap_counts':intersections,'total_utterances':sum(x['utterances'] for x in reports),'total_duration_hours':round(sum(x['duration_hours'] for x in reports),6)}
    if set(TRAIN_SPLITS).issubset(splits):
        combined = manifests/'train-960.jsonl'
        temporary = combined.with_suffix('.jsonl.tmp')
        with temporary.open('wb') as out:
            for split in TRAIN_SPLITS:
                with (manifests/(split+'.jsonl')).open('rb') as source:
                    shutil.copyfileobj(source,out,1024*1024)
        temporary.replace(combined)
        training = [r for r in reports if r['split'] in TRAIN_SPLITS]
        report['training_union'] = {'manifest':combined.relative_to(ROOT).as_posix(),'source_splits':list(TRAIN_SPLITS),'utterances':sum(r['utterances'] for r in training),'duration_hours':round(sum(r['duration_hours'] for r in training),6),'speakers':len(set().union(*(speakers[s] for s in TRAIN_SPLITS))),'includes_dev_or_test':False}
    target = ROOT/'data/reports/librispeech_inventory.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    table = ['# LibriSpeech Data Preparation Verification Results', '',
             f"Verification run time: {report['prepared_at_utc']} (UTC). Original: [OpenSLR SLR12](https://www.openslr.org/12), CC BY 4.0.", '',
             '| Split | Number of utterances | Number of speakers | Voice length (hours) |', '|---|---:|---:|---:|']
    table += [f"| {r['split']} | {r['utterances']:,} | {r['speakers']} | {r['duration_hours']:.3f} |" for r in reports]
    table += ['', f"Total **{report['total_utterances']:,}utterances / {report['total_duration_hours']:.3f}hours**. Original archives: {sum(x['archive_bytes'] for x in archive_records)/1e9:.3f} It's GB.", '',
              f'- selected {len(splits)}archives match their official MD5 checksums.',
              '- We read the STREAMINFO of all FLACs and checked the number and length of 16 kHz·mono·16-bit, samples.',
              '- All voices and transcription IDs correspond one-to-one, and there are no duplicate or empty transcriptions.',
              '- There is no duplication of utterance ID and speaker duplication between splits.',
              '- A decoding inspection of the entire waveform was not performed. The original payload was confirmed using the compressed file checksum.', '',
              'Originals: `data/raw/librispeech/LibriSpeech/`. Utterance manifests: `data/processed/manifests/`.', '',
              'This report verifies raw audio, transcripts and utterance metadata. Historical 100h runs remain separate; this collection did not generate new tokenizers/features/alignments or start model training.', '',
              '[Full Machine Readable Report](librispeech_inventory.json) · [Data Guide](../../docs/datasets_and_mac.md)', '']
    if 'training_union' in report:
        union = report['training_union']
        table += [f"Union of three training splits: **{union['utterances']:,}utterances / {union['duration_hours']:.3f}hours / {union['speakers']:,}speakers**. Combined in `data/processed/manifests/train-960.jsonl` with original splits preserved, excluding dev/test.", '']
    target.with_suffix('.md').write_text('\n'.join(table))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('100h','960h'),default='100h',help='Training scope; both options include all four dev/test splits')
    parser.add_argument('--mirror',choices=tuple(MIRRORS),default='primary',help='Official OpenSLR mirror; segments can be resumed across mirrors')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--connections',type=int,default=8,help='Parallel byte ranges per archive (1–16)')
    parser.add_argument('--inventory-only',action='store_true')
    args = parser.parse_args()
    splits = tuple(SPLITS) if args.scope == '960h' else DEFAULT_SPLITS
    base_url = MIRRORS[args.mirror]
    raw = ROOT/'data/raw/librispeech'
    raw.mkdir(parents=True,exist_ok=True)
    records = []
    if not args.inventory_only:
        if not shutil.which('curl'):
            parser.error('curl is required')
        checksum_path = raw/'md5sum.txt'
        fetch(base_url+'md5sum.txt',checksum_path)
        checksums = {line.split()[1]:line.split()[0] for line in checksum_path.read_text().splitlines() if line.strip()}
        with ThreadPoolExecutor(max_workers=max(1,min(args.workers,len(splits)))) as pool:
            jobs = {pool.submit(prepare_split,s,raw,checksums,args.connections,base_url):s for s in splits}
            for job in as_completed(jobs):
                records.append(job.result())
    else:
        for split in splits:
            checksum = SPLITS[split]
            archive=raw/'archives'/(split+'.tar.gz')
            if not archive.is_file() or md5(archive) != checksum:
                raise ValueError(f'{split}: a checksum-verified archive is required')
            marker = raw/('verified-'+split+'.json')
            source_url = json.loads(marker.read_text())['source_url'] if marker.is_file() else base_url+archive.name
            records.append({'split':split,'archive_md5':checksum,'archive_bytes':archive.stat().st_size,'archive_url':source_url})
    inventory(raw,ROOT/'data/processed/manifests',records,splits)


if __name__ == '__main__':
    main()
