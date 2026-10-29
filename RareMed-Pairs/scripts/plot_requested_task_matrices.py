"""Render actual task matrices; absent predictions get a NOT RUN chart."""
import json
from html import escape
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from medcpt_local import ROOT

REPORT=ROOT/'reports/gbm_astro_cadasil_v1'

def main():
    source=REPORT/'matrices.json'
    if not source.exists():raise ValueError('Run evaluate_requested_tasks.py score first')
    records=json.loads(source.read_text(encoding='utf-8'));out=REPORT/'plots';out.mkdir(parents=True,exist_ok=True)
    cards=[]
    for r in records:
        name=re.sub(r'[^A-Za-z0-9_-]','_',r['task']+'_'+r['dataset']+'_'+r['group'])
        cm=np.asarray(r['matrix'],dtype=int)
        if cm.shape!=(len(r['labels']),len(r['labels'])):raise ValueError('Matrix/label shape mismatch')
        fig,ax=plt.subplots(figsize=(7.2,6.4));fig.patch.set_facecolor('white')
        image=ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(int(cm.max()),1))
        n=int(cm.sum());nlabels=len(r['labels'])
        for i in range(nlabels):
            for j in range(nlabels):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=14,color='white' if cm[i,j]>cm.max()*.55 and cm.max()>0 else '#16334b')
        labels=r['labels'];ax.set_xticks(range(nlabels),labels,rotation=25 if nlabels>2 else 0,ha='right' if nlabels>2 else 'center')
        ax.set_yticks(range(nlabels),labels);ax.set_xlabel('Predicted');ax.set_ylabel('Reference')
        title=f'{r["dataset"]} | {r["task"]}\n{r["group"]} | {n:,} {r["unit"]}s'
        if not n:title+='\nNO SCORED RECORDS / NOT RUN'
        ax.set_title(title,fontsize=12,pad=16)
        fig.colorbar(image,ax=ax,fraction=.046,pad=.04)
        note='Rows = reference; columns = prediction.'
        if len(labels)==2:note+=' Cells: [[TN, FP], [FN, TP]].'
        fig.text(.5,.025,note,ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
        cards.append('<section><h2>'+escape(r['dataset']+' / '+r['task']+' / '+r['group'])+'</h2><p>'+escape(r['note'])+'</p><p>计数单位：'+escape(r['unit'])+'；矩阵中的关系数不等于独立病例数。</p><a href="plots/'+name+'.svg">SVG</a><br><img src="plots/'+name+'.png"></section>')
    html='<!doctype html><meta charset="utf-8"><title>分任务混淆矩阵</title><style>body{font:16px/1.7 Arial,"Microsoft YaHei";margin:30px;background:#f4f7fb}section{background:white;padding:22px;margin:25px 0;border:1px solid #ddd}img{max-width:800px;width:100%}</style><h1>分任务／数据集／病种混淆矩阵</h1><p>行是真值，列是预测。空矩阵明确标为无可评分记录。MultiCaRe病种为提及分层；诊断仅MedPix；头部MRI标签不等于精确脑MRI。原文描述的矩阵是文字对应恢复，另有Word-F1和ROUGE-L。</p>'+''.join(cards)
    (REPORT/'plots.html').write_text(html,encoding='utf-8')
    print(json.dumps({'matrix_figures':len(records),'gallery':str(REPORT/'plots.html')},indent=2))

if __name__=='__main__':main()
