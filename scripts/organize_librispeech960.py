#!/usr/bin/env python3
"""Organize verified LibriSpeech manifests and existing dev roles, offline.

Run prepare_librispeech.py --scope 960h first. Audio and transcripts are kept
unchanged. Local NeMo manifests contain absolute paths; rerun after moving
the repository/data to another machine. No tokenizer, features or training.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from prepare_librispeech import EXPECTED_UTTERANCES, SPLITS, TRAIN_SPLITS

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT/'data/processed/manifests'
PROTOCOL = ROOT/'data/processed/protocols/librispeech960_v1'
ROLES = ('model_selection', 'gate_fit', 'gate_calibration')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    temporary.replace(path)


def export_nemo(source, target, *, seen=None, train_hash=None, keep_rows=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.jsonl.tmp')
    native_hash, nemo_hash = hashlib.sha256(), hashlib.sha256()
    count, seconds, speakers, rows = 0, 0.0, set(), []
    with source.open('rb') as stream, temporary.open('wb') as output:
        for line in stream:
            native_hash.update(line)
            if train_hash is not None:
                train_hash.update(line)
            row = json.loads(line)
            if seen is not None:
                if row['utterance_id'] in seen:
                    raise ValueError(f"Duplicate utterance: {row['utterance_id']}")
                seen.add(row['utterance_id'])
            audio = (ROOT/row['audio']).resolve()
            if not audio.is_relative_to(ROOT/'data/raw/librispeech') or not audio.is_file():
                raise ValueError(f"Missing or out-of-root audio: {row['utterance_id']}")
            if row['sample_rate'] != 16000 or row['channels'] != 1 or not row['text'].strip():
                raise ValueError(f"Invalid manifest row: {row['utterance_id']}")
            if abs(row['duration_seconds']-row['num_samples']/16000) > 1e-9:
                raise ValueError(f"Duration mismatch: {row['utterance_id']}")
            entry = {'audio_filepath':str(audio), 'duration':row['duration_seconds'], 'text':row['text']}
            data = (json.dumps(entry, ensure_ascii=False)+'\n').encode()
            output.write(data)
            nemo_hash.update(data)
            count += 1
            seconds += row['duration_seconds']
            speakers.add(row['speaker_id'])
            if keep_rows:
                rows.append(row)
    temporary.replace(target)
    return {
        'manifest':source.relative_to(ROOT).as_posix(),
        'nemo_manifest':target.relative_to(ROOT).as_posix(),
        'utterances':count, 'speakers':len(speakers),
        'duration_hours':round(seconds/3600, 6),
        'sha256':native_hash.hexdigest(), 'nemo_sha256':nemo_hash.hexdigest(),
    }, rows


def main():
    inventory = json.loads((ROOT/'data/reports/librispeech_inventory.json').read_text())
    by_split = {row['split']:row for row in inventory['splits']}
    if set(by_split) != set(SPLITS) or inventory.get('training_union', {}).get('utterances') != 281241:
        raise ValueError('Complete --scope 960h preparation before organizing data')
    if any(inventory['speaker_overlap_counts'].values()):
        raise ValueError('Inventory reports speaker overlap')
    config = json.loads((ROOT/'configs/librispeech960_benchmark.json').read_text())['dataset']['dev_protocol']
    summary_path = ROOT/config['public_summary']
    public_protocol = json.loads(summary_path.read_text())
    assignment_path = ROOT/config['local_assignment']
    assignment = json.loads(assignment_path.read_text()) if assignment_path.exists() else {
        'protocol':'librispeech960_v1', 'seed':config['seed'],
        'group_by':'speaker_id', 'splits':{},
    }
    if assignment['seed'] != config['seed'] or assignment['group_by'] != 'speaker_id':
        raise ValueError('Existing dev assignment does not match the adopted protocol')
    catalog, dev_rows, seen, train_hash = {}, {}, set(), hashlib.sha256()
    for split in SPLITS:
        print(f'ORGANIZE {split}', flush=True)
        stats, rows = export_nemo(MANIFESTS/(split+'.jsonl'), MANIFESTS/'nemo'/(split+'.jsonl'),
                                  seen=seen, train_hash=train_hash if split in TRAIN_SPLITS else None,
                                  keep_rows=split.startswith('dev-'))
        if stats['utterances'] != EXPECTED_UTTERANCES[split] or abs(stats['duration_hours']-by_split[split]['duration_hours']) > 1e-6:
            raise ValueError(f'{split}: manifest differs from verified inventory')
        catalog[split] = stats
        if rows:
            dev_rows[split] = rows
    catalog['train-960'], _ = export_nemo(MANIFESTS/'train-960.jsonl', MANIFESTS/'nemo/train-960.jsonl')
    if catalog['train-960']['utterances'] != 281241 or catalog['train-960']['sha256'] != train_hash.hexdigest():
        raise ValueError('train-960 is not the exact ordered training union')
    role_rows = {role:[] for role in ROLES}
    for index, split in enumerate(('dev-clean', 'dev-other')):
        rows = dev_rows[split]
        if catalog[split]['sha256'] != public_protocol['splits'][split]['source_manifest_sha256']:
            raise ValueError(f'{split}: adopted dev source manifest changed')
        if split not in assignment['splits']:
            speakers = sorted({row['speaker_id'] for row in rows}, key=int)
            random.Random(config['seed']+index).shuffle(speakers)
            groups, offset = {}, 0
            for role, size in zip(ROLES, config['speaker_counts'][split]):
                selected = set(speakers[offset:offset+size])
                offset += size
                groups[role] = {'speaker_ids':sorted(selected, key=int),
                                'utterance_ids':[r['utterance_id'] for r in rows if r['speaker_id'] in selected]}
            if offset != len(speakers):
                raise ValueError(f'{split}: partition sizes do not cover all speakers')
            assignment['splits'][split] = groups
        used_ids, used_speakers = set(), set()
        for role in ROLES:
            group = assignment['splits'][split][role]
            wanted = set(group['utterance_ids'])
            selected = [row for row in rows if row['utterance_id'] in wanted]
            speakers = {row['speaker_id'] for row in selected}
            if len(selected) != len(wanted) or len(wanted) != len(group['utterance_ids']) or speakers != set(group['speaker_ids']):
                raise ValueError(f'{split}/{role}: invalid assignment')
            if used_ids & wanted or used_speakers & speakers:
                raise ValueError(f'{split}/{role}: dev role overlap')
            used_ids |= wanted
            used_speakers |= speakers
            name = f'{split}.{role}'
            destination = PROTOCOL/'manifests'/(name+'.jsonl')
            write_rows(destination, selected)
            catalog[name], _ = export_nemo(destination, PROTOCOL/'nemo'/(name+'.jsonl'))
            expected = public_protocol['splits'][split]['partitions'][role]
            if catalog[name]['utterances'] != expected['utterances'] or len(speakers) != expected['speakers']:
                raise ValueError(f'{split}/{role}: differs from adopted dev protocol')
            expected['manifest'] = catalog[name]['manifest']
            expected['nemo_manifest'] = catalog[name]['nemo_manifest']
            role_rows[role].extend(selected)
        if used_ids != {row['utterance_id'] for row in rows}:
            raise ValueError(f'{split}: dev roles do not cover all utterances')
    for role, rows in role_rows.items():
        destination = PROTOCOL/'manifests'/(role+'.jsonl')
        write_rows(destination, rows)
        catalog[role], _ = export_nemo(destination, PROTOCOL/'nemo'/(role+'.jsonl'))
    if not assignment_path.exists():
        write_json(assignment_path, assignment)
    write_json(summary_path, public_protocol)
    report = {
        'prepared_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'inventory':'data/reports/librispeech_inventory.json',
        'dev_protocol':'data/reports/librispeech960_protocol.json',
        'checks':{'original_split_utterance_overlap':0, 'dev_role_utterance_overlap':0,
                  'dev_role_speaker_overlap':0, 'dev_roles_cover_all_dev':True,
                  'train_union_exact_concatenation':True, 'all_manifest_audio_paths_exist':True},
        'transcript_policy':'Original uppercase text preserved; no text normalization, filtering, tokenizer or features generated.',
        'nemo_format_source':'https://docs.nvidia.com/nemo/speech/latest/asr/datasets.html',
        'nemo_portability':'Local NeMo JSONL uses absolute audio paths. Rerun this script on the destination machine; native manifests use repository-relative paths.',
        'manifests':catalog,
    }
    write_json(ROOT/'data/reports/librispeech960_organization.json', report)
    print(json.dumps({'organized_manifests':len(catalog), 'nemo_manifests':len(catalog),
                      'training_utterances':catalog['train-960']['utterances'],
                      'checks':report['checks']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
