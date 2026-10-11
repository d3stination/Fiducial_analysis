"""Phase 2 entry point. Never runs the HeartCycle generation pipeline."""
from pathlib import Path
import argparse
import json
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase2_analysis import replay_sources, heartcycle_hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--use-verified-cache',action='store_true',help='Rendering development only; final validation uses fresh raw replay')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    work=ROOT/'work/phase2'; work.mkdir(parents=True,exist_ok=True)
    snapshot=work/'heartcycle_before.json'
    before=heartcycle_hashes()
    if snapshot.exists():
        inherited=json.loads(snapshot.read_text())
        inherited={path:digest for path,digest in inherited.items() if path.startswith('results/tables/') and Path(path).stem.split('_',1)[1] in {
            'master_beats','detection_status','valid_detections','failed_detections','reference_eligibility','reference_errors','participant_summary','morphology_features'}}
        if inherited!=before: raise RuntimeError('HeartCycle source-table preservation baseline differs')
    else: snapshot.write_text(json.dumps(before,indent=2),encoding='utf-8')
    cache=work/'verified_replay.pkl'
    if args.use_verified_cache:
        from src.morphology_analysis import _legacy_api
        from src.data_loading import load_config
        from src.phase2_analysis import sha
        _legacy_api(load_config())
        with cache.open('rb') as handle: bundle=pickle.load(handle)
        if any(sha(path)!=digest for path,digest in bundle[-1].items()):
            raise RuntimeError('Cached replay source hashes changed; run fresh raw replay')
    else:
        bundle=replay_sources()
        with cache.open('wb') as handle: pickle.dump(bundle,handle)
    if not args.audit_only:
        from src.phase2_outputs import build_outputs
        build_outputs(bundle,before)
    assert before==heartcycle_hashes(), 'HeartCycle output changed'
    print('HeartCycle source-table checksums unchanged; Phase 2 command completed.',flush=True)


if __name__=='__main__': main()
