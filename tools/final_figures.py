"""Publication figures from the frozen CSV tables; no inference or retraining."""
import os
from pathlib import Path
from files import EVIDENCE,GROUPS,ROOT
from review_metrics import read_rows
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    palette={'O2M':'#137177','O2O':'#874293','Dual':'#C66028'}
    all_epochs=read_rows(EVIDENCE/'epochs_all.csv'); valid=read_rows(EVIDENCE/'validation_all.csv')
    fig,axes=plt.subplots(2,2,figsize=(12.2,8),layout='constrained');axes=axes.flat
    for name in GROUPS:
        subset=[r for r in all_epochs if r['group']==name];xs=[int(r['epoch']) for r in subset]
        for ax,key in zip((axes[0],axes[1]),('ap5095','ap50')):
            ax.plot(xs,[100*float(r[key]) for r in subset],label=name+(' 历史 FP16' if name=='O2M' else ''),color=palette[name],linestyle='--' if name=='O2M' else '-',linewidth=1.9)
            if name=='O2M':
                patched=[r for r in valid if r['scope']=='subset250' and r['group']=='O2M']
                ax.scatter([int(r['epoch']) for r in patched],[100*float(r[key]) for r in patched],marker='D',facecolors='white',edgecolors=palette[name],zorder=5,label='A 统一重评')
        arms=['one','many'] if name=='Dual' else ['many' if name=='O2M' else 'one']
        for arm in arms:
            label=name+(' · '+('O2M' if arm=='many' else 'O2O') if name=='Dual' else '')
            axes[2].plot(xs,[sum(float(r[arm+'_'+key]) for key in ('box','cls','dfl')) for r in subset],color=palette[name],linestyle='--' if arm=='many' else '-',label=label)
        axes[3].plot(xs,[float(r['grad_backbone_mean']) for r in subset],color=palette[name],linestyle='--' if name=='O2M' else '-',label=name)
    for ax,title,unit in zip(axes,['01  AP50–95','02  AP50','03  分支训练损失（不叠加）','04  Backbone 梯度'],['AP（%）','AP（%）','加权 box + cls + DFL','每步 L2 范数的轮均值']):
        ax.set_title(title,loc='left',fontweight='bold');ax.set(xlabel='Epoch',ylabel=unit,xticks=[1,5,10,15,20]);ax.grid(axis='y',alpha=.18);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('固定 250 张验证图与配对训练：20 轮轨迹',fontsize=16,fontweight='bold');fig.savefig(ROOT/'figures/01_逐轮曲线.png',dpi=155);plt.close(fig)
    subset=[r for r in valid if r['epoch']=='20' and r['scope']=='subset250'];full=[r for r in valid if r['epoch']=='20' and r['scope']=='official_val2510']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,key,title in zip(axes,('ap5095','ap50'),('AP50–95','AP50')):
        xs=np.arange(3)
        for idx,(items,label,color) in enumerate(((subset,'固定 250 图','#87BBB3'),(full,'完整 Val · 2,510 图','#135F65'))):
            vals=[100*float(next(r for r in items if r['group']==name)[key]) for name in GROUPS];bars=ax.bar(xs+(-.19 if idx==0 else .19),vals,.36,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set(xticks=xs,xticklabels=GROUPS,ylabel='AP（%）',title=title);ax.margins(y=.22);ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.17)
    fig.suptitle('统一 PSA 评估 · E20 固定终点（完整 Val 包含子集）',fontsize=15,fontweight='bold');fig.savefig(ROOT/'figures/02_两个验证范围.png',dpi=155);plt.close(fig)


if __name__=='__main__':
    main()
