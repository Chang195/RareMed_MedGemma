"""Plot audited MedGemma results; empty templates produce marked layouts, never fake curves.

python plot_sft_results.py
python plot_sft_results.py --diagnostic diagnostic_predictions.csv --medqa medqa_item_results.csv
See README.md for the frozen protocol and normalization contract. Values in CSV are fractions.
"""
from pathlib import Path
import argparse, json, hashlib, math, os, tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'raremed-figures-mpl-cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS=['Base','A','B','C']
COLORS={'recall':'#0072B2','fpr':'#D55E00','balanced_accuracy':'#009E73','medqa_accuracy':'#CC79A7'}
def read_csv(p): return pd.read_csv(p,keep_default_na=False,encoding='utf-8-sig')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe_div(a,b): return a/b if b else np.nan
def rates(y,p):
    tp=int(np.sum((y==1)&(p==1))); fn=int(np.sum((y==1)&(p==0)))
    fp=int(np.sum((y==0)&(p==1))); tn=int(np.sum((y==0)&(p==0)))
    r=safe_div(tp,tp+fn); f=safe_div(fp,fp+tn)
    precision=safe_div(tp,tp+fp)
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,recall=r,fpr=f,specificity=1-f,
      precision=precision,F1=safe_div(2*tp,2*tp+fp+fn),balanced_accuracy=(r+1-f)/2)
def assert_fields(df,fields):
    missing=set(fields)-set(df.columns)
    if missing: raise ValueError('Missing columns: '+str(missing))
def check_models(df):
    if set(df.model)!=set(MODELS): raise ValueError('Require all four models on identical item sets.')
def ci(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]) if len(a) else (np.nan,np.nan)
def wilson(k,n):
    if not n: return (np.nan,np.nan)
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-d,mid+d
def require_protocol(manifest,kind,df):
    expected=manifest.get(kind+'_protocol_id')
    if not expected: raise ValueError('Freeze '+kind+'_protocol_id in manifest before using results.')
    if set(df.protocol_id)!={expected}: raise ValueError('Protocol ID mismatch.')
    if set(df.model_revision)!={manifest['revision']}: raise ValueError('Model revision mismatch.')
    for model in MODELS:
        if set(df.loc[df.model==model,'checkpoint'])!={manifest['models'][model]['checkpoint']}:
            raise ValueError('Checkpoint mismatch: '+model)
def diagnostic(df,reference,manifest,repeats,seed):
    assert_fields(df,['model','sample_id','y_true','y_pred','cluster_id','image_sha256','raw_output','parse_category','checkpoint','protocol_id','model_revision'])
    check_models(df); require_protocol(manifest,'diagnostic',df)
    if df.duplicated(['model','sample_id']).any(): raise ValueError('Duplicate diagnostic item/model.')
    ref=reference.set_index('sample_id'); ordered=ref.index.tolist()
    y=(ref.y_true=='CADASIL').to_numpy(dtype=int); labels={'CADASIL':1,'non-CADASIL':0}
    groups=ref.cluster_id.to_numpy(); unique=np.unique(groups)
    pure=all(len(np.unique(y[groups==g]))==1 for g in unique)
    group_indices={g:np.flatnonzero(groups==g) for g in unique}
    strata=[np.array([g for g in unique if y[group_indices[g][0]]==c]) for c in [0,1]] if pure else [unique]
    strata=[s for s in strata if len(s)]
    rng=np.random.default_rng(seed)
    draws=[np.concatenate([group_indices[g] for s in strata for g in rng.choice(s,len(s),replace=True)]) for _ in range(repeats)]
    output={}
    for model in MODELS:
        d=df[df.model==model].set_index('sample_id')
        if set(d.index)!=set(ordered): raise ValueError('Test set mismatch: '+model)
        d=d.loc[ordered]
        for field in ['y_true','cluster_id','image_sha256']:
            if not (d[field].to_numpy()==ref[field].to_numpy()).all(): raise ValueError(field+' mismatch: '+model)
        valid=d.y_pred.isin(labels); unresolved=int((~valid).sum())
        expected_category=d.y_pred.map({'CADASIL':'explicit_positive','non-CADASIL':'explicit_negative'})
        if not (d.loc[valid,'parse_category']==expected_category[valid]).all():
            raise ValueError('Resolved diagnosis requires matching explicit parse category: '+model)
        result=dict(diagnostic_n=len(d),unresolved_n=unresolved,
          diagnostic_status='COMPLETE' if unresolved==0 else 'UNRESOLVED: point estimates withheld',
          diagnostic_ci_method='stratified cluster percentile bootstrap' if pure else 'pooled cluster percentile bootstrap',
          cluster_n=len(unique),all_CADASIL_prediction_collapse=False)
        # Never silently count unresolved outputs as negative or remove them.
        if unresolved: output[model]=result; continue
        p=d.y_pred.map(labels).to_numpy(dtype=int); result.update(rates(y,p))
        result['all_CADASIL_prediction_collapse']=bool(np.all(p==1))
        boot=[rates(y[i],p[i]) for i in draws]
        for metric in ['recall','fpr','balanced_accuracy','specificity','precision','F1']:
            values=np.array([b[metric] for b in boot]); low,high=ci(values)
            result.update({metric+'_low':low,metric+'_high':high,metric+'_valid_bootstraps':int(np.isfinite(values).sum())})
        output[model]=result
    return output
def medqa(df,manifest,repeats,seed,question_reference):
    assert_fields(df,['model','question_id','question_sha256','gold_answer','predicted_answer','raw_output','checkpoint','protocol_id','dataset_revision','model_revision'])
    check_models(df); require_protocol(manifest,'medqa',df)
    revision=manifest.get('medqa_dataset_revision'); expected_n=manifest.get('medqa_expected_n')
    if not revision or not expected_n: raise ValueError('Freeze MedQA dataset revision and expected item count first.')
    if manifest.get('medqa_sft_overlap_audit_passed') is not True:
        raise ValueError('A documented MedQA/SFT item overlap audit is required.')
    if set(df.dataset_revision)!={revision}: raise ValueError('MedQA dataset revision mismatch.')
    if df.duplicated(['model','question_id']).any(): raise ValueError('Duplicate QA item/model.')
    ref=df[df.model=='Base'].set_index('question_id').sort_index()
    if len(ref)!=expected_n: raise ValueError('MedQA item count mismatch.')
    assert_fields(question_reference,['question_id','question_sha256','gold_answer'])
    if question_reference.question_id.duplicated().any(): raise ValueError('Duplicate question reference.')
    truth=question_reference.set_index('question_id')
    if set(truth.index)!=set(ref.index): raise ValueError('MedQA reference item set mismatch.')
    if not (truth.loc[ref.index,['gold_answer','question_sha256']].to_numpy()==ref[['gold_answer','question_sha256']].to_numpy()).all():
        raise ValueError('MedQA results differ from frozen question reference.')
    if ref.question_sha256.eq('').any(): raise ValueError('Missing question hash.')
    correct={}; results={}
    for model in MODELS:
        d=df[df.model==model].set_index('question_id')
        if set(d.index)!=set(ref.index): raise ValueError('MedQA item set mismatch.')
        d=d.loc[ref.index]
        if not (d[['gold_answer','question_sha256']].to_numpy()==ref[['gold_answer','question_sha256']].to_numpy()).all():
            raise ValueError('MedQA gold/question mismatch.')
        if not d.gold_answer.isin(list('ABCD')).all(): raise ValueError('Expected normalized four-option MedQA labels A-D.')
        # Invalid/missing QA answer = incorrect, retained in denominator.
        c=(d.predicted_answer==d.gold_answer).to_numpy(dtype=int); correct[model]=c
        low,high=wilson(int(c.sum()),len(c))
        results[model]=dict(medqa_status='COMPLETE',medqa_n=len(c),medqa_correct=int(c.sum()),
            medqa_invalid_n=int((~d.predicted_answer.isin(list('ABCD'))).sum()),medqa_accuracy=c.mean(),
            medqa_accuracy_low=low,medqa_accuracy_high=high,medqa_ci_method='Wilson 95% by question')
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(ref),(repeats,len(ref)))
    for model in MODELS:
        delta=correct[model]-correct['Base']; low,high=ci(delta[draws].mean(axis=1))
        results[model].update(medqa_delta=delta.mean(),medqa_delta_low=low,medqa_delta_high=high,
          medqa_improved=int(np.sum(delta==1)),medqa_regressed=int(np.sum(delta==-1)),
          medqa_delta_ci_method='paired question percentile bootstrap')
    return results
def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
      'font.size':12,'axes.labelsize':12,'axes.titlesize':14,'axes.linewidth':.9,
      'xtick.labelsize':11,'ytick.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
      'savefig.facecolor':'white','figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
def setup(ax,letter,title,ylabel,scatter=False):
    ax.set_title(title,loc='left',pad=20,fontweight='bold')
    ax.text(-.13,1.08,letter,transform=ax.transAxes,fontsize=17,fontweight='bold')
    ax.set_ylim(-2,102); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel(ylabel)
    if scatter:
        ax.set_xlim(-2,102); ax.set_xticks([0,25,50,75,100]); ax.set_xlabel('CADASIL Balanced Accuracy (%)')
    else:
        ax.set_xlim(12,88); ax.set_xticks([20,50,80]); ax.set_xlabel('CADASIL fraction in SFT training (%)',labelpad=10)
    ax.tick_params(direction='out',length=4); ax.grid(axis='y',color='#E6E8EB',linewidth=.6); ax.set_axisbelow(True)
def has(s,m): return m in s and s[m].notna().all()
def pending(ax,text):
    ax.text(.5,.55,text,ha='center',va='center',transform=ax.transAxes,color='#6B7280',fontsize=13,linespacing=1.7)
def metric_panel(ax,s,key,color):
    if not has(s,key): pending(ax,'RESULTS PENDING\nNo estimate or 95% CI'); return
    arms=s.loc[['A','B','C']]; x=100*arms.cadasil_fraction.to_numpy(float); y=100*arms[key].to_numpy(float)
    lo=100*arms[key+'_low'].to_numpy(float); hi=100*arms[key+'_high'].to_numpy(float)
    ax.errorbar(x,y,yerr=np.vstack([np.maximum(y-lo,0),np.maximum(hi-y,0)]),fmt='o-',color=color,lw=1.7,ms=6,capsize=4)
    for a,xx,yy in zip('ABC',x,y): ax.annotate(a,(xx,yy),xytext=(6,7),textcoords='offset points',fontsize=11)
    b=s.loc['Base',key]*100; ax.axhline(b,color='#444444',lw=1.2,ls=(0,(4,3)))
    # Base CI shown independently at right edge, never at 0% SFT.
    bl=s.loc['Base',key+'_low']*100; bh=s.loc['Base',key+'_high']*100
    ax.errorbar([86],[b],yerr=[[max(b-bl,0)],[max(bh-b,0)]],fmt='s',color='#444444',capsize=3,ms=4)
def export(fig,out,stem):
    for ext in ['pdf','svg','png']: fig.savefig(out/(stem+'.'+ext),dpi=300)
    plt.close(fig)
def figures(s,out,manifest):
    style(); complete=has(s,'recall') and has(s,'medqa_accuracy')
    label='ILLUSTRATIVE LAYOUT — RESULTS PENDING' if not complete else 'Exploratory retrospective comparison · 95% confidence intervals'
    f,axes=plt.subplots(1,3,figsize=(16,7.2)); f.subplots_adjust(left=.075,right=.975,bottom=.27,top=.72,wspace=.38)
    f.suptitle('Effect of CADASIL Supervision Density on Diagnostic Performance',x=.055,ha='left',y=.96,fontsize=19,fontweight='bold')
    f.text(.055,.865,label,fontsize=12,color='#A24E22' if not complete else '#555555')
    for ax,letter,title,key,ylabel in zip(axes,'ABC',['CADASIL recall','CADASIL false positive rate','Balanced accuracy'],
      ['recall','fpr','balanced_accuracy'],['CADASIL Recall (%)','False Positive Rate (%)','Balanced Accuracy (%)']):
        setup(ax,letter,title,ylabel); ax.title.set_color(COLORS[key]); metric_panel(ax,s,key,COLORS[key])
    composition='   ·   '.join(f"{m}: {int(s.loc[m,'train_CADASIL_n'])}/{int(s.loc[m,'train_n'])} CADASIL" for m in 'ABC')
    f.text(.075,.12,composition+'   |   Base: no SFT; reference pending' if not has(s,'recall') else
      'Dashed line: Base (no SFT); square + whisker at right: Base 95% CI',fontsize=11)
    f.text(.075,.065,f"Shared holdout: {manifest['test_n']} images; CADASIL {manifest['test_positive_n']} images / {manifest['test_positive_groups']} source groups. Source labels are not clinical gold standards.",fontsize=10,color='#555555')
    export(f,out,'figure_A_diagnostic_performance')
    f,axes=plt.subplots(1,2,figsize=(16,7.2)); f.subplots_adjust(left=.08,right=.97,bottom=.27,top=.70,wspace=.30)
    f.suptitle('Trade-off Between Rare-Disease Specialization\nand General Medical Knowledge Retention',x=.055,ha='left',y=.98,fontsize=19,fontweight='bold',linespacing=1.25)
    f.text(.055,.815,label,fontsize=12,color='#A24E22' if not complete else '#555555')
    setup(axes[0],'A','CADASIL diagnostic performance','Balanced Accuracy (%)'); axes[0].title.set_color(COLORS['balanced_accuracy']); metric_panel(axes[0],s,'balanced_accuracy',COLORS['balanced_accuracy'])
    setup(axes[1],'B','MedQA performance retention','MedQA Accuracy (%)'); axes[1].title.set_color(COLORS['medqa_accuracy']); metric_panel(axes[1],s,'medqa_accuracy',COLORS['medqa_accuracy'])
    if has(s,'medqa_accuracy'):
        for m in 'ABC':
            r=s.loc[m]; axes[1].annotate(f"Δ {r.medqa_delta*100:+.1f} pp",(r.cadasil_fraction*100,r.medqa_accuracy*100),xytext=(0,-23),textcoords='offset points',ha='center',fontsize=10)
    f.text(.08,.12,'Base reference / ΔMedQA: not estimable until matched question-level results are available.' if not complete else
      'Dashed lines: Base (no SFT); ΔMedQA in percentage points, based on paired questions.',fontsize=11)
    f.text(.08,.065,'All models require the same diagnostic holdout, MedQA items, prompt, generation and scoring protocol.',fontsize=10,color='#555555')
    export(f,out,'figure_B_specialization_retention')
    f,ax=plt.subplots(figsize=(10,7.2)); f.subplots_adjust(left=.15,right=.95,bottom=.23,top=.72)
    f.suptitle('Rare-disease specialization and MedQA retention',x=.06,ha='left',y=.96,fontsize=18,fontweight='bold')
    f.text(.06,.855,label,fontsize=12,color='#A24E22' if not complete else '#555555')
    setup(ax,'A','Specialization–retention space','MedQA Accuracy (%)',True)
    if complete:
        points=s[['balanced_accuracy','medqa_accuracy']].to_numpy(float)*100
        for i,m in enumerate(MODELS):
            x,y=points[i]; r=s.loc[m]
            frontier=not any(np.all(q>=points[i]) and np.any(q>points[i]) for q in points)
            ax.errorbar(x,y,xerr=[[max(x-r.balanced_accuracy_low*100,0)],[max(r.balanced_accuracy_high*100-x,0)]],
              yerr=[[max(y-r.medqa_accuracy_low*100,0)],[max(r.medqa_accuracy_high*100-y,0)]],fmt='s' if m=='Base' else 'o',
              color=['#444444','#0072B2','#009E73','#D55E00'][i],capsize=3,ms=8,mec='#222222' if frontier else 'none',mew=1.2)
            ax.annotate(m,(x,y),xytext=(8,7),textcoords='offset points')
        f.text(.15,.11,'Outline: point-estimate non-dominated condition; uncertainty may prevent dominance conclusions.',fontsize=9)
    else: pending(ax,'RESULTS PENDING\nBase / A / B / C points not yet available\nNo trade-off or Pareto frontier estimated')
    f.text(.15,.055,'CADASIL and MedQA results must both be available for the same checkpoint.',fontsize=10,color='#555555')
    export(f,out,'figure_B_pareto_scatter')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('input_manifest.json'))
    p.add_argument('--diagnostic',type=Path); p.add_argument('--medqa',type=Path)
    p.add_argument('--outdir',type=Path,default=Path(__file__).parent)
    p.add_argument('--bootstrap',type=int,default=10000); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    if a.bootstrap<100: p.error('Use at least 100 bootstrap repetitions.')
    manifest=json.loads(a.manifest.read_text(encoding='utf-8')); root=a.manifest.parent
    dpath=a.diagnostic or root/manifest['diagnostic_results']; qpath=a.medqa or root/manifest['medqa_results']
    d=read_csv(dpath); q=read_csv(qpath); reference=read_csv(root/manifest['reference_file'])
    metrics={m:dict(model=m,train_n=v['train_n'],train_CADASIL_n=v['positive_n'],train_non_CADASIL_n=v['negative_n'],
      cadasil_fraction=v['fraction'],checkpoint=v['checkpoint'],model_id=manifest['model_id'],revision=manifest['revision'],
      planned_test_n=manifest['test_n'],planned_test_positive_n=manifest['test_positive_n'],planned_test_negative_n=manifest['test_negative_n'],
      diagnostic_status='MISSING PREDICTIONS',medqa_status='MISSING ITEM RESULTS') for m,v in manifest['models'].items()}
    if len(d):
        for m,r in diagnostic(d,reference,manifest,a.bootstrap,a.seed).items(): metrics[m].update(r)
    if len(q):
        qref=read_csv(root/manifest.get('medqa_reference_file','templates/medqa_reference.csv'))
        for m,r in medqa(q,manifest,a.bootstrap,a.seed,qref).items(): metrics[m].update(r)
    s=pd.DataFrame(metrics.values()).set_index('model').loc[MODELS]
    # Keep expected numerical fields blank when evidence is absent, never use zero.
    for key in ['TP','FN','FP','TN','recall','fpr','specificity','precision','F1','balanced_accuracy','medqa_accuracy','medqa_delta']:
        for col in ([key,key+'_low',key+'_high'] if key not in ['TP','FN','FP','TN'] else [key]):
            if col not in s: s[col]=np.nan
    s.to_csv(a.outdir/'summary_metrics.csv',encoding='utf-8-sig',na_rep='')
    provenance=dict(manifest_sha256=digest(a.manifest),diagnostic_input_sha256=digest(dpath),medqa_input_sha256=digest(qpath),
      script_sha256=digest(__file__),bootstrap_repeats=a.bootstrap,seed=a.seed,
      numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
      illustrative=not(has(s,'recall') and has(s,'medqa_accuracy')))
    (a.outdir/'plot_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    figures(s,a.outdir,manifest); print(s[['train_n','cadasil_fraction','diagnostic_status','medqa_status']].to_string())
if __name__=='__main__': main()
