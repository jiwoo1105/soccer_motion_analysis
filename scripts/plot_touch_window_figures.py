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
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
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
    """Paper overview in the original monochrome, four-stage layout.

    Landmarks supply the metric stage as a whole. Touch frames select only
    head and shoulder windows; raw ball coordinates never enter either metric.
    """
    fig, ax = plt.subplots(figsize=(8, 11.1))
    ax.set(xlim=(-8, 808), ylim=(1110, -8))
    ax.axis('off')

    def text(x, y, value, size=9, weight='normal', ha='center'):
        ax.text(x, y, value, fontsize=size, fontweight=weight,
                ha=ha, va='center', color='black', linespacing=1.35)

    def box(x, y, w, h, title, body='', fill='white', dashed=False,
            rounded=False):
        kw = dict(facecolor=fill, edgecolor='black', linewidth=.8,
                  linestyle=(0, (2.5, 2.5)) if dashed else '-')
        patch = (FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0,rounding_size=13', **kw)
                 if rounded else Rectangle((x, y), w, h, **kw))
        ax.add_patch(patch)
        text(x+w/2, y+h/2-8 if body else y+h/2, title,
             weight='bold' if rounded or fill != 'white' else 'normal')
        if body:
            text(x+w/2, y+h/2+9, body)

    def group(x, y, w, h, title):
        ax.add_patch(Rectangle((x, y), w, h, facecolor='none',
                     edgecolor='#555555', linewidth=.65, linestyle=(0, (8, 7))))
        text(x+8, y+21, title, weight='bold', ha='left')

    def path(points, dashed=False, arrow=True):
        ls = (0, (3, 3)) if dashed else '-'
        # A single polyline keeps the original fine, uninterrupted connectors.
        ax.plot(*zip(*points), color='black', lw=.7, linestyle=ls)
        if arrow:
            ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                         arrowstyle='-|>', mutation_scale=10,
                         color='black', lw=.7, linestyle=ls,
                         shrinkA=0, shrinkB=0))

    box(270, 0, 260, 64, '입력 영상', '(단안 카메라, 후방 시점)', rounded=True)
    group(20, 110, 320, 350, '(1) 관절 좌표 추출')
    group(460, 110, 320, 350, '(2) 공 추적')
    box(90, 160, 180, 55, 'OpenCV', '(BGR → RGB)')
    ball_title = 'YOLOv8' if research else 'YOLO11s'
    ball_detail = '(초기 공 위치 검출)' if research else '(발목 주변 ROI 공 검출)'
    box(530, 160, 180, 55, ball_title, ball_detail)
    path([(355, 64), (180, 160)])
    path([(445, 64), (620, 160)])
    box(80, 270, 200, 60, 'MediaPipe Pose', '(model_complexity=2)', fill='#d3d3d3')
    box(520, 270, 200, 60, 'SAM2' if research else '프레임별 공 후보',
        '(마스크 기반 공 추적)' if research else '(원본 좌표 복원)', fill='#d3d3d3')
    box(70, 380, 220, 60, '33개 관절 랜드마크', '(이미지 2D + 월드 3D)', dashed=True)
    box(510, 380, 220, 60, '공 궤적', '(cx, cy, 반지름/프레임)', dashed=True)
    for x in (180, 620):
        path([(x, 215), (x, 270)])
        path([(x, 330), (x, 380)])

    group(200, 510, 400, 120, '(3) 터치 시점 감지')
    box(240, 550, 320, 60, '터치 감지', '(방향 전환 + 발 근접도)', fill='#bfbfbf')
    path([(180, 440), (180, 495), (365, 495), (365, 550)], dashed=True)
    path([(620, 440), (620, 495), (480, 495), (480, 550)])

    group(0, 680, 800, 290, '(4) 평가지표 산출')
    # One input to the whole stage avoids repeating the same pose connection
    # to all three boxes. The stage subtitle makes the common input explicit.
    path([(125, 440), (125, 490), (150, 490), (150, 680)])
    text(160, 653, '관절 좌표', size=8, ha='left')
    text(18, 725, '월드 3D 관절 좌표 사용', size=8, ha='left')
    # Branch touch frames to head and shoulder only, in black dashed lines.
    path([(400, 610), (400, 760)], dashed=True)
    path([(400, 657), (650, 657), (650, 760)], dashed=True)
    text(420, 642, '터치 프레임', size=8, ha='left')

    metric_boxes = [
        (40, '1. 상체 각도', '무릎–엉덩이–어깨\n(월드 3D)\n전체 유효 프레임 평균'),
        (290, '2. 헤드업 범위', '어깨–눈 각도\n(월드 3D, Y축)\n' + r'터치 $\pm\Delta$ 프레임 범위'),
        (540, '3. 어깨 방향 변화 속도', '어깨 벡터 (월드 3D)\nSG 평활화 후 방향 변화 속도\n' + r'터치 $\pm B_{\mathrm{sh}}$ 프레임 평균'),
    ]
    for x, title, body in metric_boxes:
        ax.add_patch(Rectangle((x, 760), 220, 150, facecolor='white',
                              edgecolor='black', linewidth=.8))
        text(x+110, 794, title, weight='bold')
        text(x+110, 849, body)
    text(400, 943, r'$\Delta=16,\quad B_{\mathrm{sh}}=30$', size=9)
    path([(400, 970), (400, 1040)])
    box(270, 1040, 260, 64, '평가지표 산출값',
        r'$S_{\mathrm{trunk}},\ S_{\mathrm{head}},\ V_{\mathrm{touch}}$', rounded=True)
    fig.subplots_adjust(left=.015, right=.985, bottom=.005, top=.995)
    name = 'paper-touch-window-architecture' if research else 'touch-window-architecture'
    for ext in ('png', 'svg'):
        target = out/f'{name}.{ext}'
        fig.savefig(target, dpi=200, facecolor='white')
        if ext == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
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
