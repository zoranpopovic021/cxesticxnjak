import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import math as m
from scipy.stats import chisquare
from statistics import stdev
from scipy.stats import kstest
from scipy import stats
from scipy.stats import entropy
import simpleaudio as sa

TWOPI = 2*np.pi
bufflen = 1403
repeat = 10
A_freq = 110
G = 0
a = 400
b = 400
N = 150
min_vel=0.25
max_vel=3.5
d_coef = 0.75
b_coef = 1.5
min_d = 0.3
a_min = 1
a_max = 12
max_cnt = 2
c_top = 1.0
interval = 50 # ms
loop_len = 5.0 # seconds per loop
quit = False
clix = size = []
c_rate = 0.95
c_M = a_max*50/max_vel
min_clix = 0.05
alpha = 0.05
sample_rate = 44100
global l, x, y, vel, phi, area

def reset():
   global phi, vel, colors, area, x, y, clix, c_M, size
   #size = []
   clix = []
   phi = np.random.rand(int(N))*TWOPI
   vel = min_vel+(max_vel-min_vel)*np.random.rand(int(N))
   colors = np.random.rand(int(N))
   area = (a_min + a_max * np.random.rand(int(N)))**2  # 1 to 5 point radii
   x = np.random.rand(int(N))*a
   y = np.random.rand(int(N))*b
   flag = True
   cnt = 0
   for i in range(1,int(N)):
      cnt += 1
      while flag:
         flag = False
         for j in range(0,i-1):
            d = m.sqrt(m.pow(l.get_offsets()[i][0]-l.get_offsets()[j][0],2) + \
               m.pow(l.get_offsets()[i][1]-l.get_offsets()[j][1],2))
            if (d<=d_coef*(m.sqrt(area[i]/np.pi)+m.sqrt(area[j]/np.pi))+min_d):
               flag = cnt<100
               x[i]=np.random.rand(1)*a
               y[i]=np.random.rand(1)*b
               break

def update_slider_N(val):
    global size, is_manual, N, l, colors, area, phi, vel, x, y, max_vel, a_max, a_min, min_vel, N, quit
    #is_manual=True
    N = int(val)
    quit = False
    px = 1/plt.rcParams['figure.dpi'] 
    size = fig.get_size_inches()/px
    plt.close(fig)
#    fig.canvas.draw_idle()

def update_slider_v(val):
    global size, is_manual, N, l, colors, area, phi, vel, x, y, max_vel, a_max, a_min, min_vel, N, quit
    max_vel = val
    quit = False
    c_M = a_max*50/max_vel
    px = 1/plt.rcParams['figure.dpi'] 
    size = fig.get_size_inches()/px
    plt.close(fig)

def update_slider_a(val):
    global size, is_manual, N, l, colors, area, phi, vel, x, y, max_vel, a_max, a_min, min_vel, N, quit
    a_max = val
    quit = False
    c_M = a_max*50/max_vel
    px = 1/plt.rcParams['figure.dpi'] 
    size = fig.get_size_inches()/px
    plt.close(fig)

def update_slider_T(val):
    global is_manual, N, l, colors, area, phi, vel, x, y, max_vel, a_max, a_min, min_vel, N, quit
    E = 0.0
    for i in range(len(vel)):
       E += vel[i]*vel[i]*m.sqrt(area[i]/np.pi)/2
    E = E/N
    vel = vel*(m.sqrt(val/E))

def update_slider_cM(val):
    global is_manual, N, l, colors, area, phi, vel, x, y, max_vel, a_max, a_min, min_vel, N, quit, c_M
    c_M = val

def update_slider_r(val):
    global is_manual, N, l, colors, area, phi, vel, x, y, max_vel, a_max, a_min, min_vel, N, quit, c_rate
    c_rate = val

def collide(i, j, off):
    d = m.sqrt(m.pow(l.get_offsets()[i][0]-l.get_offsets()[j][0],2) + m.pow(l.get_offsets()[i][1]-l.get_offsets()[j][1],2))
    do = m.sqrt(m.pow(off[i][0]-off[j][0],2) + m.pow(off[i][1]-off[j][1],2))
    d = min(d,do)
    if (d<=d_coef*(m.sqrt(area[i]/np.pi)+m.sqrt(area[j]/np.pi))+min_d):
        dx = (l.get_offsets()[i][0]-l.get_offsets()[j][0])
        dy = (l.get_offsets()[i][1]-l.get_offsets()[j][1])
        dx1 = (off[i][0]-off[j][0])
        dy1 = (off[i][1]-off[j][1])
        if dx*dx1>0 and dy*dy1>0:
           dx=dx1
           dy=dy1
        vix = vel[i]*m.cos(phi[i])
        viy = vel[i]*m.sin(phi[i])
        vjx = vel[j]*m.cos(phi[j])
        vjy = vel[j]*m.sin(phi[j])
        tmp = ((vix-vjx)*dx+(viy-vjy)*dy)/(dx*dx+dy*dy)
        vix = vix - 2*area[j]/(area[i]+area[j])*tmp*dx
        viy = viy - 2*area[j]/(area[i]+area[j])*tmp*dy
        vjx = vjx + 2*area[i]/(area[i]+area[j])*tmp*dx
        vjy = vjy + 2*area[i]/(area[i]+area[j])*tmp*dy
        if (vix<0):
           phi[i] = np.pi+m.atan(viy/vix)
        else:
           phi[i] = m.atan(viy/vix)
        if phi[i]<0: phi[i]=2*np.pi+phi[i]
        if (vjx<0):
           phi[j] = np.pi+m.atan(vjy/vjx)
        else:
           phi[j] = m.atan(viy/vix)
        if phi[j]<0: phi[j]=2*np.pi+phi[j]
        vel[i] = m.sqrt(vix*vix+viy*viy)
        vel[j] = m.sqrt(vjx*vjx+vjy*vjy)
        return True
    else:
        return False

def test(data, exp):
    p_val = stat = [0]*data.shape[1]
    for i in range(data.shape[1]):
        stat[i], p_val[i] = kstest(data[:,i], stats.uniform(loc=exp[i][0], scale=exp[i][1]-exp[i][0]).cdf)
    return stat, p_val

def play(p_audio):
       global play_obj
       t = np.arange(0,bufflen/sample_rate,1/sample_rate)
       t_audio = np.array([])
       for n in range(len(p_audio)):
           t_audio = np.hstack( (t_audio, np.sin(A_freq*(2**(p_audio[n]/12))*t*2*np.pi)) )
       t_audio *= 32767 / np.max(np.abs(t_audio))
       t_audio = t_audio.astype(np.int16)
       if play_obj.is_playing():
          play_obj.stop()
       play_obj = sa.play_buffer(t_audio, 1, 2, sample_rate)

def advance():
    tmp = []
    cnt = int(N)
    for i in range(len(l.get_offsets())):
       coef = 1.0
       if not G==0:
          for j in range(len(l.get_offsets())):
              if not i==j:
                 dx = l.get_offsets()[i,0]-l.get_offsets()[j,0]
                 dy = l.get_offsets()[i,1]-l.get_offsets()[j,1]
                 d = m.sqrt(dx**2+dy**2)
                 vix = vel[i]*m.cos(phi[i])-G*m.sqrt(area[j]/np.pi)*dx/d**3
                 viy = vel[i]*m.sin(phi[i])-G*m.sqrt(area[j]/np.pi)*dy/d**3
                 vel[i] -= G*m.sqrt(area[j]/np.pi)/d
                 if (vix<0):
                    phi[i] = np.pi+m.atan(viy/vix)
                 else:
                    phi[i] = m.atan(viy/vix)
                 if phi[i]<0: phi[i]=2*np.pi+phi[i]
                 vel[i] = m.sqrt(vix*vix+viy*viy)
       if (l.get_offsets()[i][0]+2*vel[i]*np.cos(phi[i])>=a-m.sqrt(area[i]/np.pi)) or (l.get_offsets()[i][0]+2*vel[i]*np.cos(phi[i])<=m.sqrt(area[i]/np.pi)):
          l.get_offsets()[i][0] = l.get_offsets()[i][0]+vel[i]*np.cos(phi[i])
          phi[i] = np.pi-phi[i]
          if phi[i]<0: phi[i]=2*np.pi+phi[i]
          coef = b_coef
       if (l.get_offsets()[i][1]+2*vel[i]*np.sin(phi[i])>=b-m.sqrt(area[i]/np.pi)) or (l.get_offsets()[i][1]+2*vel[i]*np.sin(phi[i])<=m.sqrt(area[i]/np.pi)):
          l.get_offsets()[i][1] = l.get_offsets()[i][1]+vel[i]*np.sin(phi[i])
          phi[i] = 2*np.pi-phi[i]
          coef = b_coef
       if (l.get_offsets()[i][0]+2*vel[i]*np.cos(phi[i])>=a) or \
          (l.get_offsets()[i][0]+2*vel[i]*np.cos(phi[i])<=0) or \
          (l.get_offsets()[i][1]+2*vel[i]*np.sin(phi[i])>=b) or \
          (l.get_offsets()[i][1]+2*vel[i]*np.sin(phi[i])<=0):
              cnt -= 1
              flag = True
              while flag:
                 l.get_offsets()[i][0] = np.random.rand(1)*a
                 l.get_offsets()[i][1] = np.random.rand(1)*b
                 flag = False
                 for j in range(len(l.get_offsets())):
                    if i==j: continue
                    d = m.sqrt(m.pow(l.get_offsets()[i][0]-l.get_offsets()[j][0],2) + \
                       m.pow(l.get_offsets()[i][1]-l.get_offsets()[j][1],2))
                    if (d<=d_coef*(m.sqrt(area[i]/np.pi)+m.sqrt(area[j]/np.pi))+min_d):
                       flag = True
                       break
       tmp = tmp + [ [l.get_offsets()[i][0]+coef*np.cos(phi[i])*vel[i], \
                      l.get_offsets()[i][1]+coef*np.sin(phi[i])*vel[i]] ]
    l.set_offsets(tmp)
    t.set_text("N="+str(cnt))
    data = []
    for i in range(len(vel)):
       data += [ [vel[i], phi[i], l.get_offsets()[i,0], l.get_offsets()[i,1] ] ]
    stat, p_val = test(np.array(data), [ [min_vel, max_vel], [0,2*np.pi], [0,a], [0,b] ])
    p_value = np.array(p_val[2:]).prod()*np.var(vel)/(2*np.pi)*np.var(vel)/(np.array(vel).max())
    ent1 = entropy(vel/np.array(vel).sum())
    ent2 = entropy(phi/np.array(phi).sum())
    t2.set_text(f"p={p_value:.4f}"+f", H={ent1:.4f},{ent2:.4f}")

def update_plot(num):
    global is_manual, phi, clix, l_audio, vel
    if is_manual:
        return l # don't change
    advance()
    old = l.get_offsets()
    oldphi = phi
    advance()
    new = l.get_offsets()
    l.set_offsets(old)
    phi = oldphi
    for i in range(len(vel)):
      for cl in clix:
         dx = l.get_offsets()[i][0]-cl[0]
         dy = l.get_offsets()[i][1]-cl[1]
         d = m.sqrt(m.pow(dx,2) + m.pow(dy,2))
         vix = vel[i]*m.cos(phi[i])+dx/m.pow(d,3)*cl[2]*m.sqrt(area[i])
         viy = vel[i]*m.sin(phi[i])+dy/m.pow(d,3)*cl[2]*m.sqrt(area[i])
         if (vix<0):
            phi[i] = np.pi+m.atan(viy/vix)
         else:
            phi[i] = m.atan(viy/vix)
         vel[i] = m.sqrt(vix*vix+viy*viy)
      for j in range(i+1,len(vel)):
         if collide(i,j,new):
            cnt = 0
            while True:
               cnt += 1
               if cnt>max_cnt:
                   break
               d = m.sqrt(m.pow(l.get_offsets()[i][0]-l.get_offsets()[j][0],2) + \
                   m.pow(l.get_offsets()[i][1]-l.get_offsets()[j][1],2))
               if (d<=d_coef*(m.sqrt(area[i]/np.pi)+m.sqrt(area[j]/np.pi))+min_d):
                   l.get_offsets()[i][0] = l.get_offsets()[i][0]+2*vel[i]*np.cos(phi[i])
                   l.get_offsets()[i][1] = l.get_offsets()[i][1]+2*vel[i]*np.sin(phi[i])
                   l.get_offsets()[j][0] = l.get_offsets()[j][0]+2*vel[j]*np.cos(phi[j])
                   l.get_offsets()[j][1] = l.get_offsets()[j][1]+2*vel[j]*np.sin(phi[j])
               else:
                   break
    for cl in clix:
       cl[2] *= c_rate 
       if cl[2]<min_clix:
           clix.remove(cl)
           print("clix=",clix)
    # x = x + vel*np.cos(phi)
    # y = y + vel*np.sin(phi)
    is_manual = False # the above line called update_slider, so we need to reset this
    E = 0.0
    for i in range(len(vel)):
       E += vel[i]*vel[i]*m.sqrt(area[i]/np.pi)/2
    if len(l_audio)>repeat:
       l_audio = l_audio[1:]+[E/N]
    else:
       l_audio += [E/N]
    play(l_audio)
    samp4.set_val(E/N)
    return l,

def on_click(event):
    # Check where the click happened
    global clix
    inside = False
    for (xm,ym),(xM,yM) in [samp.label.clipbox.get_points(), samp2.label.clipbox.get_points(), \
       samp3.label.clipbox.get_points(), samp4.label.clipbox.get_points(), samp5.label.clipbox.get_points(), \
       samp6.label.clipbox.get_points()]:
       if (xm < event.x < xM and ym < event.y < yM):
           # Event happened within a slider, ignore since it is handled in update_slider
           inside = True
           break
    if not inside:
           # user clicked somewhere else on canvas = unpause
           global is_manual
           is_manual=False
           if (event.xdata is not None) and (event.ydata is not None):
              clix += [[event.xdata, event.ydata, c_M]]

def loop():
  reset()
  global G, play_obj, audio, l_audio, size, l, t, t2, fig, ax, axamp, axamp2, axamp3, samp, samp2, samp3, is_manual, scale, ani, area, vel, phi, colors, x, y, max_vel, a_max, a_min, min_vel, N, clix, samp4, samp5, samp6, c_rate, c_M
  fig, ax = plt.subplots()
  audio = [0]*(bufflen*repeat)
  audio = np.array(audio).astype(np.int16)
  l_audio = []
  play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
  if len(size)>0:
    px = 1/plt.rcParams['figure.dpi'] 
    fig.set_size_inches(size[0]*px,size[1]*px)
  l = plt.scatter(x, y, s=area, c=colors, alpha=0.5)
  ax = plt.axis([0,a,0,b])
  axamp = plt.axes([0.08, .03, 0.22, 0.02])
  t = plt.text(-125.25,2.15,"N="+str(N),fontsize=10)
  t2 = plt.text(-125.25,1.75,"M="+str(N),fontsize=10)
  axamp2 = plt.axes([0.38, .03, 0.22, 0.02])
  axamp3 = plt.axes([0.70, .03, 0.22, 0.02])
  axamp4 = plt.axes([0.08, .95, 0.22, 0.02])
  axamp5 = plt.axes([0.38, .95, 0.22, 0.02])
  axamp6 = plt.axes([0.70, 0.95, 0.22, 0.02])
# Slider
  samp = Slider(axamp, 'N', 1, 500, valinit=N)
  samp2 = Slider(axamp2, 'v', min_vel, min_vel*10, valinit=max_vel)
  samp3 = Slider(axamp3, 'm', a_min, a_min*10, valinit=a_max)
  samp4 = Slider(axamp4, 'T', 0.1, 50, valinit=a_max)
  samp5 = Slider(axamp5, 'c', 1, 500, valinit=c_M)
  samp6 = Slider(axamp6, 'r', 0, 1, valinit=c_rate)
# Animation controls
  is_manual = False # True if user has taken control of the animation
  scale = interval / 1000 / loop_len
# call update function on slider value change
  samp.on_changed(update_slider_N)
  samp2.on_changed(update_slider_v)
  samp3.on_changed(update_slider_a)
  samp4.on_changed(update_slider_T)
  samp5.on_changed(update_slider_cM)
  samp6.on_changed(update_slider_r)
  fig.canvas.mpl_connect('button_press_event', on_click)
  ani = animation.FuncAnimation(fig, update_plot, interval=interval, blit=False, cache_frame_data=False)
  return plt.show()

def main_loop():
  global quit, size
  quit = False
  while not quit:
    quit = True
    loop()
    if len(size)==0:
       px = 1/plt.rcParams['figure.dpi'] 
       size = fig.get_size_inches()/px

if __name__ == '__main__':
   main_loop()
