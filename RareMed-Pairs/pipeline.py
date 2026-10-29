"""Canonical project entry point. `list` prints final stages; API stages require opt-in."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGES={
 'search':(['collect.py','--config','configs/discovery_1000.json','--search-only'],False),
 'collect-plan':(['run_collection_v3.py','--dry-run'],False),
 'collect':(['run_collection_v3.py'],False),
 'extract-plan':(['run_deepseek_batch.py','--figures','data/processed/collection_v3_figures.jsonl','--run-id','deepseek_collection_v3','--limit','1000','--dry-run'],False),
 'extract':(['run_deepseek_batch.py','--figures','data/processed/collection_v3_figures.jsonl','--run-id','deepseek_collection_v3','--limit','1000'],True),
 'package':(['package_candidates.py','--run','collection_v3','--version','candidate-v0.4','--links','data/processed/deepseek_collection_v3_links.jsonl'],False),
 'retrieve':(['medcpt_batch_v2.py','--limit','370','--output','data/processed/medcpt_all_v2'],False),
 'diagnosis-candidates':(['build_diagnosis_candidates.py'],False),
 'split':(['prepare_diagnosis_splits.py'],False),
 'export-sft':(['export_diagnosis_sft.py','--output','data/datasets/diagnosis-sft-rebuild'],False),
 'eval-scan':(['evaluate_requested_tasks.py','scan'],False),
 'eval-prepare':(['evaluate_requested_tasks.py','prepare','--fetch-text'],False),
 'eval-infer':(['evaluate_requested_tasks.py','infer','--allow-api'],True),
 'eval-score':(['evaluate_requested_tasks.py','score'],False),
 'eval-plots':(['plot_requested_task_matrices.py'],False),
 'eval-ci':(['summarize_disease_pipeline_ci.py'],False),
 'medpix-prepare':(['medpix_glioma_eval.py','prepare'],False),
 'medpix-infer':(['medpix_glioma_eval.py','infer','--allow-api'],True),
 'medpix-score':(['medpix_glioma_eval.py','score'],False),
 'article-flow':(['plot_article_flow.py'],False),
 'dataset-flow':(['plot_dataset_flow.py'],False),
}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['list',*STAGES]);ap.add_argument('--allow-api',action='store_true');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
 if a.stage=='list':
  print(json.dumps({k:{'script':v[0][0],'paid_api':v[1]} for k,v in STAGES.items()},indent=2));return
 cmd,paid=STAGES[a.stage];command=[sys.executable,str(ROOT/'scripts'/cmd[0]),*cmd[1:]]
 if a.dry_run:print(json.dumps({'stage':a.stage,'arguments':cmd,'paid_api':paid}));return
 if paid and not a.allow_api:ap.error('This stage sends paid requests. Add --allow-api only when ready; configure DEEPSEEK_API_KEY locally.')
 raise SystemExit(subprocess.run(command,cwd=ROOT).returncode)
if __name__=='__main__':main()
