import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

st.header("Motion of particle under external force (Classical and Electrodynamical)",divider='rainbow')

st.subheader("Damping force\nThis force comes under classical mechanics where particle is subjected to an external force and under its influence it oscillates")
st.subheader("Damping\n Damped oscillation is an example of forced vibration. Forced vibration occurs when a system is subjected to an external force.There are three types of damping: critical, overdamped, and underdamped. In a critically damped system, the oscillations die out quickly. In an overdamped system, the oscillations are so slow that they might as well not be oscillating at all.")
st.image("Damping force.PNG")

st.subheader("Lorentz force\nThis force comes under electrodynamics where the particle is subjected to an external magnetic and electric field and particle moves under their influence")
st.subheader("Lorentz force\nIn physics, specifically in electromagnetism, the Lorentz force (or electromagnetic force) is the combination of electric and magnetic force on a point charge due to electromagnetic fields. A particle of charge q moving with a velocity v in an electric field E and a magnetic field B experiences a force")
st.image("lorenzforce.jpg")

plot1=st.checkbox("Damping Force")
plot2=st.checkbox("Lorentz Force")



def damp():
    st.title('DAMPING FORCE ')
    st.header('Types of damping force',divider='rainbow')

    A=int(st.number_input('Enter the Amplitude'))
    m=int(st.number_input('Enter the mass',min_value=1))
    k=int(st.number_input('Enter the spring constant value'))
    b=int(st.number_input('Enter damping constant value'))
    #dt = 0.01

    tmax = 40
    dt = 0.001

    print(b**2-4*m*k)

    def funcv(v):
        return ((1/m)*(-k*(x[i])-(b*v)))
    def funcx():
        return v[i]


    t = [0]
    v = [0]
    x = [A]
    i = 0

    while t[-1] <= tmax:
        k1x = funcx()
        k2x = funcx()
        k3x = funcx()
        k4x = funcx()

        x.append(x[i] + ((1 / 6) * (k1x + 2 * k2x + 2 * k3x + k4x) * dt))

        k1 = funcv(v[i])
        k2 = funcv(v[i] + 0.5 * k1 * dt)
        k3 = funcv(v[i] + 0.5 * k2 * dt)
        k4 = funcv(v[i] + k3 * dt)

        v.append(v[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * dt)

        t.append(t[-1] + dt)
        i += 1

    if (b**2-4*m*k)>0:
        plt.subplot(1,2,1)
        plt.plot(t,x,color='red')
        plt.ylim(-A-1, A+1)
        plt.grid()
        plt.title('Overdamped \n position vs time')
        plt.subplot(1,2,2)
        plt.plot(t,v,color='red')
        plt.grid()
        plt.title('velocity vs time')
        plt.savefig('graph.jpg')
        st.image('graph.jpg')

    if (b**2-4*m*k)==0:
        plt.subplot(1,2,1)
        plt.plot(t,x,color='red')
        plt.ylim(-A-1, A+1)
        plt.grid()
        plt.title('Critical Damping \n position vs time')
        plt.subplot(1,2,2)
        plt.plot(t,v,color='red')
        plt.grid()
        plt.title('velocity vs time')
        plt.savefig('graph.jpg')
        st.image('graph.jpg')

    if (b**2-4*m*k)<0:
        plt.subplot(1,2,1)
        plt.plot(t,x,color='red')
        plt.ylim(-A-1, A+1)
        plt.grid()
        plt.title('Underdamped \n position vs time')
        plt.subplot(1,2,2)
        plt.plot(t,v,color='red')
        plt.grid()
        plt.title('velocity vs time')
        plt.savefig('graph.jpg')
        st.image('graph.jpg')

def loren():
    st.title('LORENTZ FORCE')
    st.header('',divider='rainbow')

    col1, col2, col3 = st.columns(3)

    with col1:
        ex = st.number_input('enter Electric field in x direction')
        ey = st.number_input('enter Electric field in y direction')
        ez = st.number_input('enter Electric field in z direction')
    with col2:
        bx = st.number_input('enter Magnetic field in x direction')
        by = st.number_input('enter Magnetic field in y direction')
        bz = st.number_input('enter Magnetic field in z direction')
    with col3:
        vx = st.number_input('enter X velocity')
        vy = st.number_input('enter Y velocity')
        vz = st.number_input('enter Z velocity')

    col4, col5 = st.columns(2)
    with col4:
        m = st.number_input('enter Mass of particle', 1)
        a1=st.slider('Rotation angle Vertical',0,180)
    with col5:
        q = st.number_input('enter Charge of particle')
        b1 = st.slider('Rotation angle Horizontal',0,180)
    x = 0
    y = 0
    z = 0

    a = []
    b = []
    c = []
    t = 0
    tmax = 1
    dt = 0.01


    def funcvx(f):
        return vx + (f / m) * dt


    def funcvy(f):
        return vy + (f / m) * dt


    def funcvz(f):
        return vz + (f / m) * dt


    while t <= tmax:
        fx = q * (ex + ((vy * bz) - (vz * by)))
        fy = q * (ey - ((vx * bz) + (vz * bx)))
        fz = q * (ez + ((vx * by) - (vy * bx)))

        kvx1 = funcvx(fx)
        kvx2 = funcvx(fx + 0.5 * kvx1 * dt)
        kvx3 = funcvx(fx + 0.5 * kvx2 * dt)
        kvx4 = funcvx(fx + kvx3 * dt)

        kvy1 = funcvy(fy)
        kvy2 = funcvy(fy + 0.5 * kvy1 * dt)
        kvy3 = funcvy(fy + 0.5 * kvy2 * dt)
        kvy4 = funcvy(fy + kvy3 * dt)

        kvz1 = funcvz(fz)
        kvz2 = funcvz(fz + 0.5 * kvz1 * dt)
        kvz3 = funcvz(fz + 0.5 * kvz2 * dt)
        kvz4 = funcvz(fz + kvz3 * dt)

        vx = funcvx(fx)
        vy = funcvy(fy)
        vz = funcvz(fz)

        x = x + (1 / 6) * (kvx1 + 2 * kvx2 + 2 * kvx3 + kvx4) * dt
        y = y + (1 / 6) * (kvy1 + 2 * kvy2 + 2 * kvy3 + kvy4) * dt
        z = z + (1 / 6) * (kvz1 + 2 * kvz2 + 2 * kvz3 + kvz4) * dt

        a.append(x)
        b.append(y)
        c.append(z)

        t += dt

    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("path of charged particle under influence of electric and magnetic field")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot3D(a, b, c, color='red', label='path')
    ax.legend(loc='lower left')
    ax.view_init(a1, b1)
    plt.savefig('graph1.jpg')
    st.image('graph1.jpg')

if plot1:
    damp()
if plot2:
    loren()
