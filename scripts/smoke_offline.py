"""One-epoch synthetic CPU smoke; creates no persistent data or checkpoints."""
from pathlib import Path
import os,json,subprocess,sys,tempfile
import torch as th
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'dywa/src'),str(root/'dywa/exp/train')]
from train.source_lineage import TRAIN_SOURCE_FILES,sha256_file,canonical_json_sha256
from student_FT_distill import load_FT_frozen,build_FT_rollout_targets
th.manual_seed(3)
with tempfile.TemporaryDirectory(prefix='nonprehensile-synthetic-') as tmp:
 tmp=Path(tmp);T,E=64,2
 meta={k:'SYNTHETIC_TEST_ONLY' for k in ('teacher_checkpoint_sha256','teacher_env_stat','teacher_env_stat_sha256','icp_checkpoint','icp_checkpoint_sha256','split_file','split_sha256')}
 meta.update(artifact_class='NONPREHENSILE_SOFT_DEV',synthetic_test_only=True,repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_snapshot_sha256=canonical_json_sha256({p:sha256_file(root/p) for p in TRAIN_SOURCE_FILES}),dim_state=10,dim_act=6)
 buf={'meta':meta,'h':th.randn(T,E,10),'a':th.randn(T,E,6),'valid':th.ones(T,E,dtype=th.bool),'done':th.zeros(T,E,dtype=th.bool),'episode_id':th.zeros(T,E,dtype=th.long)}
 for key,dim in [('rel_goal_delta',9),('object_delta',13),('contact',1),('progress',1)]:buf[key]=th.rand(T,E,dim)
 th.save(buf,tmp/'buffer.pt')
 env=os.environ.copy();env.update(BUFFER_PATH=str(tmp/'buffer.pt'),EPOCHS='1',BATCH='8',K='3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 result=subprocess.run([sys.executable,str(root/'scripts/run.py'),'world-model','--output',str(tmp/'out')],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 if result.returncode:
  print(result.stdout);raise SystemExit(result.returncode)
 summary=json.loads((tmp/'out/summary.json').read_text())
 model=load_FT_frozen(str(tmp/'out/ft_best.pt'),th.device('cpu'))
 targets,_=build_FT_rollout_targets(model,buf['h'][0],th.randn(E,3,6))
 assert th.isfinite(targets).all() and targets.shape==(E,3,10)
 print(json.dumps({'synthetic_test_only':True,'epochs':summary['epochs'],'train_windows':summary['n_train_windows'],'val_windows':summary['n_val_windows'],'finite_loss':bool(th.isfinite(th.tensor(summary['best_val_total']))),'reload_and_rollout':True}))
