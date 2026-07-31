from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

SRC=Path('/mnt/data/LBA_NHB_submission_v9_blue_changes/Figure1_overview_v9_revised.png')
OUT=Path('/mnt/data/LBA_reanalysis_summary_assets/Figure1_overview_CommsPsych_revised.png')
img=Image.open(SRC).convert('RGB')
d=ImageDraw.Draw(img)

# fonts
reg='/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'
bold='/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'
ital='/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf'
if not Path(reg).exists():
    reg='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'; ital='/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'
F=lambda n: ImageFont.truetype(reg,n)
FB=lambda n: ImageFont.truetype(bold,n)
FI=lambda n: ImageFont.truetype(ital,n)

BG=(248,250,243)
PANEL=(244,249,240)
BLACK=(20,20,20)
GRAY=(90,90,90)
BLUE=(76,120,168)
GREEN=(91,158,80)
RED=(213,101,80)
PURPLE=(171,111,163)
LIGHTBOX=(246,248,238)

# helpers
def center_text(box,text,font,fill=BLACK,spacing=3):
    x1,y1,x2,y2=box
    bbox=d.multiline_textbbox((0,0),text,font=font,spacing=spacing,align='center')
    w=bbox[2]-bbox[0]; h=bbox[3]-bbox[1]
    d.multiline_text(((x1+x2-w)/2,(y1+y2-h)/2),text,font=font,fill=fill,spacing=spacing,align='center')

def rounded_box(box,fill=LIGHTBOX,outline=(90,90,90),radius=7,width=1):
    d.rounded_rectangle(box,radius=radius,fill=fill,outline=outline,width=width)

# a: clinical label
box=(79,54,220,154); d.rectangle(box,fill=BG)
rounded_box((84,57,216,151),fill=(252,239,239),outline=(225,140,140),radius=4)
center_text((90,61,210,99),'Clinical\nstates',FB(14))
# retain conceptual icon by simple clinical heart motif
# heart-ish ECG line
d.line((126,126,138,126,143,115,149,137,155,121,162,126,174,126),fill=(205,79,79),width=3)

# generic stats boxes
def replace_stats(box, lines):
    d.rectangle(box,fill=BG)
    rounded_box(box,fill=(246,249,237),outline=(70,70,70),radius=7,width=1)
    x1,y1,x2,y2=box
    total=len(lines)
    line_h=(y2-y1-12)/total
    for i,(txt,font) in enumerate(lines):
        center_text((x1+4,y1+6+i*line_h,x2-4,y1+6+(i+1)*line_h),txt,font)

replace_stats((916,430,1077,519),[("n = 7,491",FB(13)),("Δ = 0.1172",F(14)),("95% CI [0.1109, 0.1238]",F(10)),("P = 0.0002",FI(13))])
replace_stats((437,901,540,995),[("n = 15",FB(12)),("Δ = 0.5636",F(13)),("95% CI",F(9)),("[0.4502, 0.6836]",F(9)),("P = 0.0002",FI(12))])
replace_stats((962,902,1084,995),[("n = 30",FB(12)),("Δ = 0.0200",F(13)),("95% CI",F(9)),("[0.0031, 0.0382]",F(9)),("P = 0.0002",FI(12))])

# CASE exclusion statement
ex=(584,795,753,886)
d.rectangle(ex,fill=BG)
d.rounded_rectangle(ex,radius=3,fill=(249,250,244),outline=(150,150,150),width=1)
center_text((588,800,749,840),'Same-subject and\nsame-video pairs excluded',FB(11))
center_text((588,840,749,879),'(cross-participant,\ncross-video comparisons)',F(9))
# Update interpretation text to context-dependent
text_box=(972,786,1085,885); d.rectangle(text_box,fill=BG)
center_text(text_box,'Same-quadrant\npairs are modestly\nmore similar, but\nthe effect is\nstimulus-sensitive.',FI(11))

# Clear and redraw lower panels e/f
bottom=(10,989,1094,1370)
d.rectangle(bottom,fill=BG)
# panel borders
# outer/bisect
for xy in [(10,989,1094,1370),(10,989,553,1370),(553,989,1094,1370)]:
    d.rectangle(xy,outline=(80,80,80),width=1)
# titles
# panel labels
d.text((27,1005),'e',font=FB(18),fill=BLACK); d.text((59,1006),'Independent-unit evidence',font=FB(16),fill=BLACK)
d.text((568,1005),'f',font=FB(18),fill=BLACK); d.text((600,1006),'Interpretation',font=FB(16),fill=BLACK)

# panel e chart
cx1,cy1,cx2,cy2=86,1068,519,1310
# axes
d.line((cx1,cy2,cx2,cy2),fill=BLACK,width=1); d.line((cx1,cy1,cx1,cy2),fill=BLACK,width=1)
ymax=0.72
for val in [0,0.2,0.4,0.6]:
    y=cy2-(val/ymax)*(cy2-cy1)
    d.line((cx1-5,y,cx1,y),fill=BLACK,width=1)
    d.text((43,y-8),f'{val:.1f}',font=F(10),fill=BLACK)
    if val>0: d.line((cx1,y,cx2,y),fill=(220,224,216),width=1)
# y label rotated via temp image
lab=Image.new('RGBA',(200,26),(0,0,0,0)); ld=ImageDraw.Draw(lab); ld.text((0,0),'Mean independent-unit Δ',font=F(12),fill=BLACK)
lab=lab.rotate(90,expand=True); img.paste(lab,(18,1110),lab)
vals=[('PTB-XL',0.117202,0.110918,0.123754,BLUE,'clinical'),('WESAD',0.563650,0.450168,0.683592,GREEN,'stress-related'),('CASE',0.020014,0.003096,0.038214,RED,'affective')]
xs=[156,300,444]
barw=54
for x,(name,val,lo,hi,col,sub) in zip(xs,vals):
    top=cy2-(val/ymax)*(cy2-cy1)
    d.rectangle((x-barw//2,top,x+barw//2,cy2),fill=col,outline=(50,50,50),width=1)
    ylo=cy2-(lo/ymax)*(cy2-cy1); yhi=cy2-(hi/ymax)*(cy2-cy1)
    d.line((x,yhi,x,ylo),fill=BLACK,width=2); d.line((x-6,yhi,x+6,yhi),fill=BLACK,width=2); d.line((x-6,ylo,x+6,ylo),fill=BLACK,width=2)
    d.text((x-28,top-22),f'{val:.4f}',font=F(10),fill=BLACK)
    center_text((x-58,1318,x+58,1355),f'{name}\n({sub})',F(10))
# chart note
center_text((70,1350,530,1367),'Error bars show participant-/patient-level cluster-bootstrap 95% CIs.',FI(9),fill=GRAY)

# panel f: interpretation cards
# legend
center_text((590,1037,1068,1070),'Independent participants or patients—not overlapping pairs—define inference.',FB(11))
card_y1,card_y2=1084,1240
cards=[(581,739,'WESAD','Strong and consistent','All 15 participant\ncontrasts were positive.',GREEN),
       (758,916,'CASE','Modest and context-dependent','Primary effect positive;\nvideo-centred effect null.',RED),
       (935,1085,'PTB-XL','Robust clinical generalization','One ECG per patient;\nage/sex sensitivity stable.',BLUE)]
for x1,x2,name,headline,body,col in cards:
    rounded_box((x1,card_y1,x2,card_y2),fill=(255,255,255),outline=(100,100,100),radius=8,width=1)
    d.rectangle((x1+1,card_y1+1,x2-1,card_y1+27),fill=col)
    center_text((x1+3,card_y1+2,x2-3,card_y1+27),name,FB(13),fill=(255,255,255))
    center_text((x1+7,card_y1+35,x2-7,card_y1+77),headline,FB(11))
    center_text((x1+7,card_y1+78,x2-7,card_y2-8),body,F(9),fill=GRAY)
# bottom conclusion
rounded_box((596,1273,1071,1350),fill=(252,246,222),outline=(166,137,79),radius=8,width=1)
center_text((606,1280,1061,1343),'Shared states leave measurable physiological similarity,\nbut its magnitude and context dependence differ across domains.',FB(13))

img.save(OUT,dpi=(400,400))
print(OUT)
