"""Scientific figures from saved source-frame signals; no model inference or fit.

Usage: python scripts/plot_touch_window_figures.py --study PATH --output docs/images
The local study's sources.json points to private originals, never uploaded here.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import font_manager


def style():
    font = Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'svg.fonttype': 'path'})


def architecture(out, research=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis('off')
    blue = '#245b91'
    def box(x, y, w, h, text, c='#222222'):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=c, lw=1.1))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=c, linespacing=1.65)
    def arrow(points, color='#222222', style='-'):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1,ls=style)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,
                                    color=color,lw=1,linestyle=style))
    box(3.3,9.1,3.4,.65,'입력 영상\n단안 카메라, 후방 촬영')
    box(.8,7.45,3.3,1.05,'관절 좌표 추출\nOpenCV → MediaPipe Pose\nimage / world landmarks')
    ball='공 추적\nYOLOv8 초기 검출 → SAM2\n공 중심 좌표' if research else '공 검출\nYOLO11s, 발목 주변 ROI\n원본 좌표의 공 후보'
    box(5.9,7.45,3.3,1.05,ball)
    arrow([(4,9.1),(2.45,8.5)]);arrow([(6,9.1),(7.55,8.5)])
    box(5.9,5.75,3.3,1.0,'터치 시점 감지\n공의 수평 방향 전환 + 발 근접\n'+r'$t_k$')
    arrow([(7.55,7.45),(7.55,6.75)])
    arrow([(4.1,7.75),(4.8,7.75),(4.8,6.25),(5.9,6.25)])
    ax.text(4.83,6.8,'발목',fontsize=9,ha='left')
    # Pose bus supplies all three metrics, independent of the touch bus.
    arrow([(2.45,7.45),(2.45,4.65),(1.6,4.65),(1.6,3.45)])
    arrow([(2.45,4.65),(4.9,4.65),(4.9,3.45)])
    arrow([(4.9,4.65),(8.2,4.65),(8.2,3.45)])
    ax.text(2.6,4.78,'관절 좌표',fontsize=10)
    # Touch bus terminates at head and shoulder, never trunk.
    arrow([(7.55,5.75),(7.55,5.2),(9.5,5.2),(9.5,2.9),(9.3,2.9)],blue)
    arrow([(7.55,5.2),(5.25,5.2),(5.25,3.45)],blue)
    ax.text(5.3,5.32,'터치 시점 → 구간 선택',fontsize=10,color=blue)
    box(.25,1.75,2.7,1.7,'상체 각도\n좌우 무릎–엉덩이–어깨\n유효 프레임 평균\n'+r'$S_{\mathrm{trunk}}$')
    box(3.55,1.75,2.7,1.7,'헤드업 범위\n머리 벡터와 수직축의 각도\n'+r'$t_k\pm\Delta$'+' 내 범위 평균\n'+r'$S_{\mathrm{head}}$')
    box(6.6,1.75,2.7,1.7,'어깨 방향 변화 속도\nSG 평활화한 3D 단위 벡터\n'+r'$t_k\pm B_{\mathrm{sh}}$'+' 내 속도 평균\n'+r'$V_{\mathrm{touch}}$',blue)
    box(3.15,.2,3.7,.75,'항목별 원시값 → 점수 환산\n고정 가중합 / 측정 불가 시 N/A')
    for x in (1.6,4.9,8.0):
        arrow([(x,1.75),(x,1.28),(5,1.28),(5,.95)])
    fig.tight_layout(pad=.4)
    name='paper-touch-window-architecture' if research else 'touch-window-architecture'
    for ext in ('png','svg'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def example(study, out, event_set="sam2_automatic", half=30):
    rows=json.loads((study/'results.json').read_text())
    sources=json.loads((study/'sources.json').read_text())
    z=np.load(study/'signals.npz',allow_pickle=False)
    alias='기준1';center=119
    src=next(x for x in sources if x['alias']==alias)
    row=next(x for x in rows if x['alias']==alias and x['event_set']==event_set and x['halfwidth']==half)
    fig=plt.figure(figsize=(11,7.7),layout='constrained')
    gs=fig.add_gridspec(3,3,height_ratios=[1.7,1.45,.25])
    cap=cv2.VideoCapture(src['source_video']);provenance=[]
    for j,f in enumerate([center-half,center,center+half]):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
        h,w=im.shape[:2];lm=z[alias+'::lm'][f];xy=lm[[0,11,12,23,24,25,26,27,28],:2]*[w,h]
        xy=xy[np.isfinite(xy).all(1)]
        lo=np.maximum(np.floor(xy.min(0)-[85,105]).astype(int),[0,0])
        hi=np.minimum(np.ceil(xy.max(0)+[85,90]).astype(int),[w,h])
        x0,y0=lo;x1,y1=hi
        ax=fig.add_subplot(gs[0,j]);ax.imshow(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB))
        sh=lm[[11,12],:2]*[w,h]-lo
        ax.annotate('',xy=sh[1],xytext=sh[0],arrowprops={'arrowstyle':'->','color':'#da3131','lw':2})
        ax.axis('off');ax.set_title(f'({chr(97+j)}) f{f}'+('  터치 후보' if j==1 else '  구간 경계'),fontsize=11)
        provenance.append({'video':alias,'frame':f,'crop':[int(a) for a in [x0,y0,x1,y1]]})
    cap.release()
    ax=fig.add_subplot(gs[1,:]);v=z[alias+'::speed'];mask=z[alias+'::'+event_set+'::'+str(half)];x=np.arange(len(v))
    ax.plot(x,v,color='#b5bac0',lw=1.3,label='전체 유효 속도')
    ax.plot(x,np.where(mask,v,np.nan),color='#245b91',lw=2,label=f'터치 ±{half} 내 유효 속도')
    for c in row['centers']:
        ax.axvspan(max(0,c-half),min(len(v)-1,c+half),color='#245b91',alpha=.065)
        ax.axvline(c,color='#777',lw=.8,ls=':')
    ax.axhline(row['mean_speed'],color='#245b91',ls='--',lw=1.1,label=f'선택 평균 {row["mean_speed"]:.2f}°/s')
    ax.set(xlabel='원본 프레임 번호',ylabel='어깨 방향 변화 속도 (°/s)',xlim=(0,len(v)-1),ylim=(0,None))
    ax.legend(ncol=3,loc='upper left',fontsize=10,frameon=False);ax.grid(axis='y',alpha=.15)
    a=fig.add_subplot(gs[2,:]);a.axis('off')
    a.text(0,.65,f'10점 영상의 예. 유효 인접 쌍 {row["valid_pairs"]}개를 한 번씩 평균함. 회색 구간은 집계에서 제외함.',fontsize=10)
    fig.savefig(out/'touch-window-example.png',dpi=180);plt.close(fig)
    (out/'touch-window-figure-provenance.json').write_text(json.dumps({'source':event_set,'alias':alias,'frames':provenance,'centers':row['centers'],'halfwidth':half,'valid_pairs':row['valid_pairs'],'mean_speed_deg_s':row['mean_speed']},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);style();architecture(a.output,True);example(a.study,a.output)
