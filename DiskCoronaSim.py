import numpy as np
import scipy
import sys
import os
import math
from math import e
from astropy import constants as const
from astropy import units as u
from scipy.optimize import fsolve
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")
import json

#----------------------------------------------------------------------
#-------------Set the initial parameters and constants-----------------
#----------------------------------------------------------------------
const_h=const.h.cgs/u.erg/u.s
const_c=const.c.cgs/u.cm*u.s
const_kb=const.k_B.cgs/u.erg*u.K
const_sigma=const.sigma_sb.cgs*(u.K)**4*u.s**3/u.g
const_a=4*const_sigma/const_c
erg_to_ev=1.6021772e12
mh=1.6733e-24
Rs_nomass=2*const.G.cgs*const.M_sun.cgs/(const.c.cgs**2)/u.cm
#The mass will be added in the loop

X_hfraction=0.7 #same taken for the OP tables, X=0.7 Z=0.2
kes=0.2*(1+X_hfraction)
Leddc=1.3e38

#from https://ui.adsabs.harvard.edu/abs/2005ApJ...620...59S/abstract
#where spin here is intended the dimensionless BH specific angular momentum |J|c/(GM^2), typically from -0.998 to 0.998
#returns the isco per unit mass, in units of rg
def compute_ISCO(spin) :
    zed1=1+(1-spin**2)**(1/3)*((1+spin)**(1/3)+(1-spin)**(1/3))
    zed2=np.sqrt(3*spin**2+zed1**2)
    if spin>0 :
        return 3+zed2-np.sqrt((3-zed1)*(3+zed1+2*zed2))
    else :
        return 3+zed2+np.sqrt((3-zed1)*(3+zed1+2*zed2))

def compute_efficiency(spin, isco) :
    energy_at_isco=(isco**2-2*isco+spin*np.sqrt(isco))/isco/np.sqrt(isco**2-3*isco+2*spin*np.sqrt(isco))
    return 1-energy_at_isco

def toiter(v):
    try:
        return list(v)
    except Exception:
        return (v,)

def runmodel(mass_input, mdot_input, photon_index=1.9,spin_input=0,
                disk_emfreq=const.c.cgs/(3000*u.angstrom).cgs*u.s, cor_emfreq=2.,
                Ex_low=2., Ex_up=10,
                mu=0.5, fmax=0.99, alpha_0=0.02,
                albedo=0.1, downward_scattering=0.55,
                overwrite=False) :
    """
    Some initial comments

    mass_input, mdot_input: Linear values are needed in the calculations. Values are normalized to solar masses
                            and Eddington, respectively. Example: 1e8 for mass and 0.5 for mdot.
                            Also solar-mass BHs can be calculated.
                            Accretion rate in the sweet-spot of (0.0x-1) Edd.
                            Use your own knowledge to give values for which a
                            geometrically-thin and optically-thick disk can
                            be used.

    photon_index: slope of the observed X-ray spectrum.

    spin_input: from -0.998 to 0.998. It will automatically change the r_ISCO and
                the radiative efficiency. If the interpolation of f(r) does funny
                things at the minimum (small r), increase jmax, the log-step of
                the radius array

    disk_emfreq: Input frequency for the disk luminosity (in units of Hz).
        Also X-ray frequency in case of a binary black hole are

    cor_emfreq: Input energy for the corona emission (in keV).
        The code will convert it to Hz.

    Ex_low, Ex_up: energy range (keV) in which a broadband mock coronal
                    luminosity is calculated. This luminosity would only
                    include the primary continuum of the Comptonizing corona.

    mu, fmax, alpha_0: model unknowns that need to be fixed.
                       See Arcodia et al. (2019).

    albedo: disk albedo fixed as in Arcodia et al. (2019).

    downward_scattering: fraction of power in the corona that is not emitted
                         towards the observed, but is instead emitted towards
                         the disk. It is included in the disk's energy balance.
                         See Arcodia et al. (2019).

    overwrite: simply a flag to overwrite previous run with the same main parameters:
               mass, accretion rate, photon index, spin.

    Written by Riccardo Arcodia: arcodia@mpe.mpg.de
    """
    name_data_dir='data'
    data_filename=name_data_dir+'/Data_logm%s_mdot%s_gamma_%s_spin_%s.json'  %( str(round(np.log10(mass_input),2)), str(mdot_input), str(photon_index), str(spin_input))

    if os.path.exists(data_filename) and overwrite==False :
        print(  "The model with logM={}, mdot={}, photon_index={}, spin={} was already run and stored".format(round(np.log10(mass_input),2),
                                                                                                              mdot_input,
                                                                                                              photon_index,
                                                                                                              spin_input) )
        return None

    else :
        print(  "Running model with logM={}, mdot={}, photon_index={}, spin={}...".format(round(np.log10(mass_input),2),
                                                                                       mdot_input,
                                                                                       photon_index,
                                                                                       spin_input) )

    if mdot_input<0.01 or mdot_input>1 :
        print(  "WARNING: you are using a value of mdot={}, not recommended given the assumptions in the model.".format(mdot_input))
        print(  "Use at your own risk. Please consider using a value between 0.0x and 1.")

    if spin_input>0.998 and spin_input<-0.998 :
        print(  "ERROR: you are using a value of spin={}, outside of the possible range (-0.998,0.998).".format(spin_input))
        return None

    #-----------------------Loading opacity tables--------------------------------------
    T_OP,rho_OP,k_OP, deriv_logk_asf_logT=np.genfromtxt('Opacity/OP_table',usecols=(0,1,3,4), comments='#', unpack=True)
    """
        Values in the OP table are all in log scale.
        The step in logT and logRho is 0.01.
        The functions yield the closest opacity value, in linear scale, in the table.
        If a 9.999 is found (i.e. the required value is outside of the table)
        it goes to the next rho for fixed T.

    """
    opacity_threshold=0.05

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]

    def Opacity_closest(T,rho) :
        #returning k in linear scale
        T_near=find_nearest(T_OP,np.log10(T))
        rho_near=find_nearest(rho_OP,np.log10(rho))
        index_of_table=np.where((T_OP==T_near)&(rho_OP==rho_near))[0]
        for index in range(index_of_table[0], len(k_OP)) :
            if k_OP[index]!=9.999 :
                return [10**k_OP[index]]

    def Derivative_Opacity_closest(T,rho) :
        #returning k in linear scale
        T_near=find_nearest(T_OP,np.log10(T))
        rho_near=find_nearest(rho_OP,np.log10(rho))
        index_of_table=np.where((T_OP==T_near)&(rho_OP==rho_near))[0]
        for index in range(index_of_table[0], len(deriv_logk_asf_logT)) :
            if deriv_logk_asf_logT[index]!=9.999 :
                return [deriv_logk_asf_logT[index]]

    #------------------Defining some input dependent quantities--------------
    eps0=compute_efficiency(spin_input, compute_ISCO(spin_input))
    r_0=compute_ISCO(spin_input)/2 #i want it in units of Rs
    log_r0=np.log10(r_0)
    if mass_input<5e9 :
        jmax=200+800*(1+abs(spin_input))**4 #Log step N between r_0 and rmax
    else :
        jmax=600+1000*(1+abs(spin_input))**4
    #Note:jmax needs to be a function of spin just for numerical problems in solving f(r) for high-spin, high_mdot and high_mass. Needed to avoid weird interpolations.
    rmax=2000
    log_rmax=np.log10(rmax)
    r = np.logspace(log_r0, log_rmax, jmax)

    b=1.
    k_1=np.sqrt(2*alpha_0/(fmax**2))
    k_0=2*b*k_1/3

    #no-torque inner boundary is assumed
    def J(r) :
        return 1.-np.sqrt(r_0/r)

    rhs_q=(3.*const_c**4*Leddc)/(64.*np.pi*(const.G.cgs*u.g*u.s*u.s/(u.cm**3))**2*(const.M_sun.cgs/u.g)**2*eps0)

    #Input frequency for the disk has to be in Hz. Default is 3000A in Hz.
    nu_disk=disk_emfreq

    #input energy/frequency for the corona.
    def keV_to_Hz(keV) :
        xray_ev=keV*1e3
        xray_erg=xray_ev*(erg_to_ev)**(-1)
        return xray_erg/const_h


    nu_x=keV_to_Hz(cor_emfreq)
    nux_low=keV_to_Hz(Ex_low)
    nux_up=keV_to_Hz(Ex_up)

    #The model will compute a bolometric X-ray luminosity
    #These are the energy limits fixed for the X-ray power-law spectrum, to extract, e.g., L2keV
    nu_x_1=2.417990504024e+16 #0.1keV
    nu_x_2=2.417990504024e+19 #100keV

    #------------------Prad>> and Pgas>> constants definition--------------
    #Prad>> regime constants
    A_const=(2**(7./2)*const_a**2*eps0*np.pi*(const.G.cgs*u.g*u.s*u.s/(u.cm**3))*(const.M_sun.cgs/u.g))/(9.*kes*Leddc*rhs_q)
    B_const=2./3*rhs_q*(3./const_a)**(3./2)
    C_const=const_a*mh/6/const_kb
    h_propconst=3*Leddc*const_c**3/(2**(9./2)*eps0*np.pi*(const.G.cgs*u.g*u.s*u.s/(u.cm**3))**2*(const.M_sun.cgs/u.g)**2*const_a)/A_const/(B_const)**2
    #Pgas>> regime constants
    xi=1.00*(k_0**(-1./3))
    T_constgas=((3*mh*Leddc*rhs_q*kes)/(2**(9./2)*np.pi*const_kb*eps0*const_a*(const.G.cgs*u.g*u.s*u.s/(u.cm**3))*(const.M_sun.cgs/u.g)))**(1./5)
    h_constgas=T_constgas**(-2.)*2**(7./4)*rhs_q*const_c**(-3.)*(kes*(const.G.cgs*u.g*u.s*u.s/(u.cm**3))*(const.M_sun.cgs/u.g)/const_a)**(1./2)
    rho_constgas=const_a*const_c**3/(6*rhs_q*kes*(const.G.cgs*u.g*u.s*u.s/(u.cm**3))*(const.M_sun.cgs/u.g))/h_constgas*T_constgas**4
    P_constgas=2*const_kb/mh*rho_constgas*T_constgas
    closure_constgas=mh*const_a/(6*const_kb*rho_constgas)*T_constgas**3
    #they contain kes (electron scattering opacity), it will be updated later in the loop

    #---------------------------------------------------------------------
    #-------------------Some computations that can be done before the loop
    #---------------------------------------------------------------------
    A_constant_terms=(A_const**(-9./(mu+4.)))*(B_const**(-10./(mu+4.)))*(C_const**(4./(mu+4.)))*(k_0**(1./(mu+4.)))*((alpha_0)** (1./(mu+4.)))
    A_radial_terms=((J(r))**(8./(mu+4.)))*(r**(-21./(2.*(mu+4.))))
    def A_mmdot(x,y) :
        return ((x)**(1./(mu+4.)))*((y)**(8./(mu+4.)))

    Q_radial_terms=((J(r))*(r**(-3.)))
    def Q_mmdot(x,y) :
        return ((x)**(-1.))*y*rhs_q

    T_nof_constant_terms=(A_const**((1-2*mu)/(mu+4.)))*(B_const**(2*(mu**2-3*mu+2)/((2-mu)*(mu+4.))))*(C_const**(mu/(mu+4.)))*(k_0**(-1./(mu+4.)))*((alpha_0)**(-1./(mu+4.)))
    T_nof_radial_terms=((J(r))**(2*mu/(mu+4.)))*(r**(3*(2*mu**2-3*mu-2)/(2*(2-mu)*(mu+4.))))
    def T_nof_mmdot(x,y) :
        return ((x)**(-1./(mu+4.)))*((y)**(2*mu/(mu+4.)))

    rho_constant_terms_rad=(A_const**(6*(2-mu)/(mu+4.)))*(B_const**(2*(8-3*mu)/(mu+4.)))*(C_const**(4*mu/(mu+4.)))*(k_0**(-4./(mu+4.)))*(alpha_0)**(-4./(mu+4.))
    rho_radial_terms_rad=(J(r)**(2*(3*mu-4)/(mu+4.)))*(r**(3*(2-3*mu)/(mu+4.)))
    def rho_mmdot_rad(x,y) :
        return (x**(-4./(mu+4.)))*(y**(2*(3*mu-4)/(mu+4.)))

    P_constant_terms_rad=const_a/3*(A_const**(4*(1-2*mu)/(mu+4.)))*(B_const**(8*(mu**2-3*mu+2)/((2.-mu)*(mu+4.))))*(C_const**(4*mu/(mu+4.)))*(k_0**(-4./(mu+4.)))*(alpha_0)**(-4./(mu+4.))
    P_radial_terms_rad=(J(r)**(8*mu/(mu+4.)))*(r**(6*(2*mu**2-3*mu-2)/((2.-mu)*(mu+4.))))
    def P_mmdot_rad(x,y) :
        return (x**(-4./(mu+4.)))*(y**(8*mu/(mu+4.)))

    rho_constant_terms_gas=rho_constgas*(k_0**(-3./5))*xi**(3./10)*(alpha_0)**(-7./10)
    rho_radial_terms_gas=(J(r))**(2./5)*(r**(-33./20))
    def rho_mmdot_gas(x,y) :
        return (x**(-7./10))*(y**(2./5))

    P_constant_terms_gas=P_constgas*(k_0**(-13./15))*xi**(1./10)*(alpha_0)**(-9./10)
    P_radial_terms_gas=(J(r))**(4./5)*(r**(-51./20))
    def P_mmdot_gas(x,y) :
        return (x**(-9./10))*(y**(4./5))

    h_constant_terms_gas=h_constgas*(k_0**(-2./15))*xi**(-1./10)*(alpha_0**(-1./10))
    h_radial_terms_gas=((J(r))**(1./5))*(r**(21./20))
    def h_mmdot_gas(x,y) :
        return (x**(-1./10))*(y**(1./5))

    def func_Lxmono(x) :
        if photon_index!=2 :
            return x*(1-(photon_index-1))*nu_x**(-(photon_index-1))/(nu_x_2**(1-(photon_index-1))-nu_x_1**(1-(photon_index-1)))
        else :
            return x*(1-(photon_index+0.0001-1))*nu_x**(-(photon_index+0.0001-1))/(nu_x_2**(1-(photon_index+0.0001-1))-nu_x_1**(1-(photon_index+0.0001-1)))

    def func_Lxbroad(x) :
        if photon_index!=2 :
            return x*(nux_up**(1-(photon_index-1))-nux_low**(1-(photon_index-1)))/(nu_x_2**(1-(photon_index-1))-nu_x_1**(1-(photon_index-1)))
        else :
            return x*(nux_up**(1-(photon_index+0.0001-1))-nux_low**(1-(photon_index+0.0001-1)))/(nu_x_2**(1-(photon_index+0.0001-1))-nu_x_1**(1-(photon_index+0.0001-1)))

    A_gas_constant_terms=(closure_constgas)*(k_0**(-1./5))*((alpha_0)**(1./10))*(xi**(-9./10))
    A_gas_radial_terms=J(r)**(4./5)*(r**(-21./20))
    def A_gas_mmdot(x,y) :
        return ((x)**(1./(10)))*((y)**(4./(5.)))

    T_gas_nof_constant_terms=T_constgas*(k_0**(-4./15))*xi**(-1./5)*((alpha_0)**(-1./5))
    T_gas_nof_radial_terms=((J(r))**(2./5))*(r**(-9./10))
    def T_gas_nof_mmdot(x,y) :
        return ((x)**(-1./(5)))*((y)**(2./5))

    Temperature_term_rad_constants=((4./3)/(kes*h_propconst*(A_const**(6*(2-mu)/(mu+4.)))*(B_const**(2*(8-3*mu)/(mu+4.)))*(C_const**(4*mu/(mu+4.)))*(k_0**(-4./(mu+4.)))*alpha_0**(-4./(mu+4.))))**(1./4)
    Temperature_term_rad_radial=(J(r)*(J(r))**(2*(3*mu-4)/(mu+4.))*(r**(3*(2-3*mu)/(mu+4.))))**(-1./4)
    def Temperature_term_rad_mmdot(x,y) :
        return (y*(x)**(-4./(mu+4.))*(y)**(2*(3*mu-4)/(mu+4.)))**(-1./4)

    Temperature_term_gas_constants=((4./3)/(kes*xi**(-1)*h_constgas*(k_0**(-7./15))*xi**(-1./10)*(alpha_0**(-1./10))*rho_constgas*(k_0**(-3./5))*xi**(3./10)*alpha_0**(-7./10)))**(1./4)
    Temperature_term_gas_radial=(((J(r))**(1./5))*(r**(21./20))*(J(r)**(2./5))*(r**(-33./20)))**(-1./4)
    def Temperature_term_gas_mmdot(x,y) :
        return ((x)**(-1./10)*(y)**(1./5)*(x)**(-7./10)*(y)**(2./5))**(-1./4)

    def BB_Flux_mononu(T) :
        return np.pi*((2*const_h*nu_disk**3/const_c**2)*(e**((const_h*nu_disk)/(const_kb*T))-1.)**(-1.))

    def Func_closure_f(x) :
        return (((2.*alpha_0)**(1./mu)-(x*k_1)**(2./mu))/((x*k_1)**(2./mu)*(1-x*(1-downward_scattering*(1-albedo)))**(9./(mu+4))))

    def Func_closure_f_gas(x) :
        return (((2.*alpha_0)**(1./mu)-(x*k_1)**(2./mu))/((x*k_1)**(2./mu)*(1-x*(1-downward_scattering*(1-albedo)))**(9./10)))

    A=[]
    Q=[]
    T_nof=[]
    A_gas=[]
    T_gas_nof=[]
    rho_nof=[]
    rho_gas_nof=[]
    P_nof=[]
    P_gas_nof=[]
    h_nof=[]
    h_gas_nof=[]
    sol_rad=[]
    radius_rad=[]
    Lx_rad=[]
    Lx_broadband_rad=[]
    Ldisk_rad=[]
    Tem_rad=[]
    Tmid_rad=[]
    rho_rad=[]
    rho_gas=[]
    P_rad=[]
    P_gas=[]
    h_rad=[]
    h_gas=[]
    radius_Ldisk_rad=[]
    sol_gas=[]
    radius_gas=[]
    Lx_gas=[]
    Lx_broadband_gas=[]
    Ldisk_gas=[]
    Tem_gas=[]
    Tmid_gas=[]
    radius_Ldisk_gas=[]
    Opacity=[]
    Qplus=[]
    Derivative_Opacity=[]
    f_array=[]

    def use_computed_stuff() :
        if check_rad==True :
            sol_rad.append(sol_rad_temp)
            f_array.append(sol_rad_temp)
            radius_rad.append(r[j])
            Lx_rad.append(Lx_rad_temp)
            Lx_broadband_rad.append(Lx_broadband_rad_temp)
            Ldisk_rad.append(Ldisk_rad_temp)
            Tem_rad.append(Tem_rad_temp)
            Tmid_rad.append(Tmid_rad_temp)
            rho_rad.append(rho_rad_temp)
            P_rad.append(P_rad_temp)
            h_rad.append(h_rad_temp)
            radius_Ldisk_rad.append(r[j])
            Opacity.append(opacity_temp)
            Qplus.append(Qplus_temp)
            Derivative_Opacity.append(Derivative_Opacity_closest(Tmid_rad_temp, rho_rad_temp)[0])
            A.append(A_temp)
            Q.append(Q_temp)
            T_nof.append(T_nof_temp)
            rho_nof.append(rho_nof_temp)
            P_nof.append(P_nof_temp)
            h_nof.append(h_nof_temp)
        elif check_rad==False :
            A_gas.append(A_gas_temp)
            Q.append(Q_temp)
            T_gas_nof.append(T_gas_nof_temp)
            rho_gas_nof.append(rho_gas_nof_temp)
            P_gas_nof.append(P_gas_nof_temp)
            h_gas_nof.append(h_gas_nof_temp)
            sol_gas.append(sol_gas_temp)
            f_array.append(sol_gas_temp)
            radius_gas.append(r[j])
            Lx_gas.append(Lx_gas_temp)
            Lx_broadband_gas.append(Lx_broadband_gas_temp)
            Ldisk_gas.append(Ldisk_gas_temp)
            Tem_gas.append(Tem_gas_temp)
            Tmid_gas.append(Tmid_gas_temp)
            radius_Ldisk_gas.append(r[j])
            P_gas.append(P_gas_temp)
            h_gas.append(h_gas_temp)
            rho_gas.append(rho_gas_temp)
            Opacity.append(opacity_temp)
            Qplus.append(Qplus_temp)
            Derivative_Opacity.append(Derivative_Opacity_closest(Tmid_gas_temp, rho_gas_temp)[0])

    def use_computed_stuff_minus1_rad() :
        if check_rad==True :
            sol_rad.append(sol_rad[-1])
            f_array.append(sol_rad[-1])
            radius_rad.append(r[j])
            Lx_rad.append(Lx_rad[-1])
            Lx_broadband_rad.append(Lx_broadband_rad[-1])
            Ldisk_rad.append(Ldisk_rad[-1])
            Tem_rad.append(Tem_rad[-1])
            Tmid_rad.append(Tmid_rad[-1])
            rho_rad.append(rho_rad[-1])
            P_rad.append(P_rad[-1])
            h_rad.append(h_rad[-1])
            radius_Ldisk_rad.append(r[j])
            Opacity.append(Opacity[-1])
            Qplus.append(Qplus[-1])
            Derivative_Opacity.append(Derivative_Opacity_closest(Tmid_rad[-1], rho_rad[-1])[0])
            A.append(A[-1])
            Q.append(Q[-1])
            T_nof.append(T_nof[-1])
            rho_nof.append(rho_nof[-1])
            P_nof.append(P_nof[-1])
            h_nof.append(h_nof[-1])
        elif check_rad==False :
            A_gas.append(A[-1])
            Q.append(Q[-1])
            T_gas_nof.append(T_nof[-1])
            rho_gas_nof.append(rho_nof[-1])
            P_gas_nof.append(P_nof[-1])
            h_gas_nof.append(h_nof[-1])
            sol_gas.append(sol_rad[-1])
            f_array.append(sol_rad[-1])
            radius_gas.append(r[j])
            Lx_gas.append(Lx_rad[-1])
            Lx_broadband_gas.append(Lx_broadband_rad[-1])
            Ldisk_gas.append(Ldisk_rad[-1])
            Tem_gas.append(Tem_rad[-1])
            Tmid_gas.append(Tmid_rad[-1])
            radius_Ldisk_gas.append(r[j])
            P_gas.append(P_rad[-1])
            h_gas.append(h_rad[-1])
            rho_gas.append(rho_rad[-1])
            Opacity.append(Opacity[-1])
            Qplus.append(Qplus[-1])
            Derivative_Opacity.append(Derivative_Opacity_closest(Tmid_rad[-1], rho_rad[-1])[0])

    def use_computed_stuff_minus1_gas() :
        if check_rad==True :
            sol_rad.append(sol_gas[-1])
            f_array.append(sol_gas[-1])
            radius_rad.append(r[j])
            Lx_rad.append(Lx_gas[-1])
            Lx_broadband_rad.append(Lx_broadband_gas[-1])
            Ldisk_rad.append(Ldisk_gas[-1])
            Tem_rad.append(Tem_gas[-1])
            Tmid_rad.append(Tmid_gas[-1])
            rho_rad.append(rho_gas[-1])
            P_rad.append(P_gas[-1])
            h_rad.append(h_gas[-1])
            radius_Ldisk_rad.append(r[j])
            Opacity.append(Opacity[-1])
            Qplus.append(Qplus[-1])
            Derivative_Opacity.append(Derivative_Opacity_closest(Tmid_gas[-1], rho_gas[-1])[0])
            A.append(A_gas[1][-1])
            Q.append(Q[-1])
            T_nof.append(T_gas_nof[-1])
            rho_nof.append(rho_gas_nof[-1])
            P_nof.append(P_gas_nof[-1])
            h_nof.append(h_gas_nof[-1])
        elif check_rad==False :
            A_gas.append(A_gas[-1])
            Q.append(Q[-1])
            T_gas_nof.append(T_gas_nof[-1])
            rho_gas_nof.append(rho_gas_nof[-1])
            P_gas_nof.append(P_gas_nof[-1])
            h_gas_nof.append(h_gas_nof[-1])
            sol_gas.append(sol_gas[-1])
            f_array.append(sol_gas[-1])
            radius_gas.append(r[j])
            Lx_gas.append(Lx_gas[-1])
            Lx_broadband_gas.append(Lx_broadband_gas[-1])
            Ldisk_gas.append(Ldisk_gas[-1])
            Tem_gas.append(Tem_gas[-1])
            Tmid_gas.append(Tmid_gas[-1])
            radius_Ldisk_gas.append(r[j])
            P_gas.append(P_gas[-1])
            h_gas.append(h_gas[-1])
            rho_gas.append(rho_gas[-1])
            Opacity.append(Opacity[-1])
            Qplus.append(Qplus[-1])
            Derivative_Opacity.append(Derivative_Opacity_closest(Tmid_gas[-1], rho_gas[-1])[0])

    #---------------------------------------------------------------------------------
    #-----------------Some computations that can be done outside the radial loop------
    #---------------------------------------------------------------------------------
    Rs_mmdot=Rs_nomass*mass_input
    A_mmdot_terms=A_mmdot(mass_input,mdot_input)
    Q_mmdot_terms=Q_mmdot(mass_input,mdot_input)
    T_nof_mmdot_terms=T_nof_mmdot(mass_input,mdot_input)
    rho_mmdot_terms_rad=rho_mmdot_rad(mass_input,mdot_input)
    rho_mmdot_terms_gas=rho_mmdot_gas(mass_input,mdot_input)
    P_mmdot_terms_rad=P_mmdot_rad(mass_input,mdot_input)
    P_mmdot_terms_gas=P_mmdot_gas(mass_input,mdot_input)
    h_mmdot_terms_gas=h_mmdot_gas(mass_input,mdot_input)
    A_gas_mmdot_terms=A_gas_mmdot(mass_input,mdot_input)
    T_gas_nof_mmdot_terms=T_gas_nof_mmdot(mass_input,mdot_input)
    Temperature_term_rad_mmdot_terms=Rs_mmdot**(-1./4)*Temperature_term_rad_mmdot(mass_input,mdot_input)
    Temperature_term_gas_mmdot_terms=Rs_mmdot**(-1./4)*Temperature_term_gas_mmdot(mass_input,mdot_input)
    for j in range (1, len(r)):
        #the numerical way obtaining "correct" opacities is not very pythonic but it kind of works so..
        #below I correct for any numerical "errors" with an interpolation
        stuck_at_first_radius=False
        opacity_jumped=False
        opacity_list_check=[]
        opacity_from_tables=kes
        iteration=0
        iteration_2=0
        while True:
            #---------------------------------------------------------------------------------
            #------------------Prad>> regime solutions
            #---------------------------------------------------------------------------------
            A_temp=A_constant_terms*(opacity_from_tables/kes)**(9./(mu+4))*A_mmdot_terms*A_radial_terms[j]
            Q_temp=Q_mmdot_terms*Q_radial_terms[j]
            T_nof_temp=T_nof_mmdot_terms*T_nof_radial_terms[j]*T_nof_constant_terms*(opacity_from_tables/kes)**((2*mu-1)/(mu+4))
            rho_nof_temp=rho_mmdot_terms_rad*rho_radial_terms_rad[j]*rho_constant_terms_rad*(opacity_from_tables/kes)**(6*(mu-2)/(mu+4))
            P_nof_temp=P_mmdot_terms_rad*P_radial_terms_rad[j]*P_constant_terms_rad*(opacity_from_tables/kes)**(4*(2*mu-1)/(mu+4))
            h_nof_temp=h_propconst*(opacity_from_tables/kes)*mdot_input*J(r[j])
            if mu!=0.0 :
                def func_solrad(x):
                    return Func_closure_f(x)-A_temp
                x0=np.zeros(10)+0.0001
                sol_fsolve=fsolve(func_solrad,x0)
                sol_real_fsolve=np.array(list(set(np.around(np.array(sol_fsolve),10))))
                x1=np.zeros(10)+0.9999
                sol_fsolve2=fsolve(func_solrad,x1)
                sol_real_fsolve=sol_real_fsolve.tolist()
                for i in sol_fsolve2 :
                    if round(i,5) not in sol_real_fsolve :
                        sol_real_fsolve.append(i)
                sol_real_fsolve=np.array(sol_real_fsolve)
                for i in range(0, len(sol_real_fsolve)):
                    #condition to have 0<f<1, Prad/Pgas>0, Prad/Pgas>1 since we are in Prad>> regime, and also a check that fsolve really found the root of the eq..
                    if (sol_real_fsolve[i]<1) and (sol_real_fsolve[i]>0) and ((2.*alpha_0)**(1./mu)-(sol_real_fsolve[i]*k_1)**(2./mu)>0) and (A_temp*(1-sol_real_fsolve[i]*(1-downward_scattering*(1-albedo)))**(9./(mu+4)))>1. and (Func_closure_f(sol_real_fsolve[i])-A_temp>-1e-5 and Func_closure_f(sol_real_fsolve[i])-A_temp<1e-5) :
                        sol_rad_temp=sol_real_fsolve[i]
                        radius_rad_temp=r[j]
                        Lxbol=sol_real_fsolve[i]*(1-downward_scattering)*Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                        Lx_rad_temp=func_Lxmono(Lxbol)
                        Lx_broadband_rad_temp=func_Lxbroad(Lxbol)
                        Temperature_term_rad=T_nof_temp*(1-sol_real_fsolve[i]*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_rad_constants*Temperature_term_rad_mmdot_terms*Temperature_term_rad_radial[j]*(opacity_from_tables/kes)**((1-2*mu)/(mu+4))
                        rho_rad_temp=rho_nof_temp*(1-sol_real_fsolve[i]*(1-downward_scattering*(1-albedo)))**(6*(mu-2)/(mu+4))
                        P_rad_temp=P_nof_temp*(1-sol_real_fsolve[i]*(1-downward_scattering*(1-albedo)))**(4*(2*mu-1)/(mu+4))
                        h_rad_temp=h_nof_temp*(1-sol_real_fsolve[i]*(1-downward_scattering*(1-albedo)))
                        Tem_rad_temp=Temperature_term_rad
                        Tmid_rad_temp=T_nof_temp*(1-sol_real_fsolve[i]*(1-downward_scattering*(1-albedo)))**((2*mu-1)/(mu+4))
                        Ldisk_rad_temp=2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)*BB_Flux_mononu(Temperature_term_rad)
                        radius_Ldisk_rad_temp=r[j]
                        opacity_temp=Opacity_closest(Tmid_rad_temp, rho_rad_temp)[0]
                        Qplus_temp=Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                        check_rad=True
            elif mu==0.0 :
                if (A_temp*(1-f*(1-downward_scattering*(1-albedo)))**(9./(mu+4)))>1. :
                    sol_rad_temp=f
                    radius_rad_temp=r[j]
                    Lxbol=f*(1-downward_scattering)*Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                    Lx_rad_temp=func_Lxmono(Lxbol)
                    Lx_broadband_rad_temp=func_Lxbroad(Lxbol)
                    Temperature_term_rad=T_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_rad_constants*Temperature_term_rad_mmdot_terms*Temperature_term_rad_radial[j]*(opacity_from_tables/kes)**((1-2*mu)/(mu+4))
                    rho_rad_temp=rho_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(6*(mu-2)/(mu+4))
                    P_rad_temp=P_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(4*(2*mu-1)/(mu+4))
                    h_rad_temp=h_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))
                    Tem_rad_temp=Temperature_term_rad
                    Tmid_rad_temp=T_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**((2*mu-1)/(mu+4))
                    Ldisk_rad_temp=2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)*BB_Flux_mononu(Temperature_term_rad)
                    radius_Ldisk_rad_temp=r[j]
                    opacity_temp=Opacity_closest(Tmid_rad_temp, rho_rad_temp)[0]
                    Qplus_temp=Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                    check_rad=True
            #---------------------------------------------------------------------------------
            #------------------Pgas>> regime solutions
            #---------------------------------------------------------------------------------
            A_gas_temp=A_gas_constant_terms*A_gas_mmdot_terms*A_gas_radial_terms[j]*(opacity_from_tables/kes)**(9./10)
            T_gas_nof_temp=T_gas_nof_mmdot_terms*T_gas_nof_radial_terms[j]*T_gas_nof_constant_terms*(opacity_from_tables/kes)**(1./5)
            rho_gas_nof_temp=rho_mmdot_terms_gas*rho_radial_terms_gas[j]*rho_constant_terms_gas*(opacity_from_tables/kes)**(-3./10)
            P_gas_nof_temp=P_mmdot_terms_gas*P_radial_terms_gas[j]*P_constant_terms_gas*(opacity_from_tables/kes)**(-1./10)
            h_gas_nof_temp=h_mmdot_terms_gas*h_radial_terms_gas[j]*h_constant_terms_gas*(opacity_from_tables/kes)**(1./10)
            if mu!=0.0 :
                def func_solgas(x):
                    return Func_closure_f_gas(x)-A_gas_temp
                x02=np.zeros(10)+0.0001
                sol_fsolve_gas=fsolve(func_solgas,x02)
                sol_real_fsolve_gas=np.array(list(set(np.around(np.array(sol_fsolve_gas),10))))
                for i in range(0, len(sol_real_fsolve_gas)):
                    if (sol_real_fsolve_gas[i]<1) and (sol_real_fsolve_gas[i]>0) and ((2.*alpha_0)**(1./mu)-(sol_real_fsolve_gas[i]*k_1)**(2./mu)>0) and (A_gas_temp*(1-sol_real_fsolve_gas[i]*(1-downward_scattering*(1-albedo)))**(9./10))<1. and (Func_closure_f_gas(sol_real_fsolve_gas[i])-A_gas_temp>-1e-5 and Func_closure_f_gas(sol_real_fsolve_gas[i])-A_gas_temp<1e-5) :
                        sol_gas_temp=sol_real_fsolve_gas[i]
                        radius_gas_temp=r[j]
                        Lxbol=sol_real_fsolve_gas[i]*(1-downward_scattering)*Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                        Lx_gas_temp=func_Lxmono(Lxbol)
                        Lx_broadband_gas_temp=func_Lxbroad(Lxbol)
                        Temperature_term_gas=T_gas_nof_temp*(1-sol_real_fsolve_gas[i]*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_gas_constants*Temperature_term_gas_mmdot_terms*Temperature_term_gas_radial[j]*(opacity_from_tables/kes)**(-1./5)
                        Tem_gas_temp=Temperature_term_gas
                        Tmid_gas_temp=T_gas_nof_temp*(1-sol_real_fsolve_gas[i]*(1-downward_scattering*(1-albedo)))**(1./5)
                        rho_gas_temp=rho_gas_nof_temp*(1-sol_real_fsolve_gas[i]*(1-downward_scattering*(1-albedo)))**(-3./10)
                        P_gas_temp=P_gas_nof_temp*(1-sol_real_fsolve_gas[i]*(1-downward_scattering*(1-albedo)))**(-1./10)
                        h_gas_temp=h_gas_nof_temp*(1-sol_real_fsolve_gas[i]*(1-downward_scattering*(1-albedo)))**(1./10)
                        Ldisk_gas_temp=2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)*BB_Flux_mononu(Temperature_term_gas)
                        radius_Ldisk_gas_temp=r[j]
                        opacity_temp=Opacity_closest(Tmid_gas_temp, rho_gas_temp)[0]
                        Qplus_temp=Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                        check_rad=False
            elif mu==0 :
                if (A_gas_temp*(1-f*(1-downward_scattering*(1-albedo)))**(9./10))<1. :
                    sol_gas_temp=f
                    radius_gas_temp=r[j]
                    Lxbol=f*(1-downward_scattering)*Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                    Lx_gas_temp=func_Lxmono(Lxbol)
                    Lx_broadband_gas_temp=func_Lxbroad(Lxbol)
                    Temperature_term_gas=T_gas_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_gas_constants*Temperature_term_gas_mmdot_terms*Temperature_term_gas_radial[j]*(opacity_from_tables/kes)**(-1./5)
                    Tem_gas_temp=Temperature_term_gas
                    Tmid_gas_temp=T_gas_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(1./5)
                    rho_gas_temp=rho_gas_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(-3./10)
                    P_gas_temp=P_gas_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(-1./10)
                    h_gas_temp=h_gas_nof_temp*(1-f*(1-downward_scattering*(1-albedo)))**(1./10)
                    Ldisk_gas_temp=2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)*BB_Flux_mononu(Temperature_term_gas)
                    radius_Ldisk_gas_temp=r[j]
                    opacity_temp=Opacity_closest(Tmid_gas_temp, rho_gas_temp)[0]
                    Qplus_temp=Q_temp*2*np.pi*r[j]*(r[j]-r[j-1])*(Rs_mmdot**2)
                    check_rad=False
            #------------------Check the opacity value and choose the next one----------------
            #I need to iteratively run the model up to an opacity/temperature/density self-consistent with the stellar tables.
            #These bit is really messy. But the interpolation below should solve possible numerical errors.
            ratio_opacities=round(opacity_temp/opacity_from_tables,3)
            if len(opacity_list_check)<3 :
                opacity_list_check.append(opacity_temp)
            elif len(opacity_list_check)==3 :
                opacity_list_check[0], opacity_list_check[1], opacity_list_check[2] = opacity_list_check[1], opacity_list_check[2], opacity_list_check[0]
                del opacity_list_check[2]
                opacity_list_check.append(opacity_temp)
            if 1-opacity_threshold<=opacity_temp/opacity_from_tables<=1+opacity_threshold :
                if len(Opacity)!=0 and (opacity_temp<0.01*Opacity[-1] or opacity_temp>100*Opacity[-1]) :
                    opacity_jumped=True
                    opacity_from_tables=Opacity[-1]
                else:
                    use_computed_stuff()
                    break
            elif len(opacity_list_check)==3 and opacity_list_check[2]==opacity_list_check[0] :
                opacity_from_tables=np.random.uniform(low=min(opacity_list_check[2], opacity_list_check[1]), high=max(opacity_list_check[2], opacity_list_check[1]))
            else :
                opacity_from_tables=(abs(opacity_temp+opacity_from_tables))/2
            if opacity_jumped==False and iteration_2==6 and len(Opacity)>0 : #safety check after a while to avoid a loop
                if r[j-1] in radius_rad :
                    use_computed_stuff_minus1_rad()
                elif r[j-1] in radius_gas :
                    use_computed_stuff_minus1_gas()
                break
            elif opacity_jumped==True and iteration_2==10 and len(Opacity)>0 : #safety check after a while to avoid a loop
                if r[j-1] in radius_rad :
                    use_computed_stuff_minus1_rad()
                elif r[j-1] in radius_gas :
                    use_computed_stuff_minus1_gas()
                break
            elif iteration_2==6 and len(Opacity)==0 :
                stuck_at_first_radius=True
            if iteration_2==10 and len(Opacity)==0 :
                use_computed_stuff()
                break
            if stuck_at_first_radius==True :
                if 0.90<=opacity_temp/opacity_from_tables<=1.10 :
                    use_computed_stuff()
                    break
            if opacity_jumped==True :
                if 0.30<=opacity_temp/opacity_from_tables<=1.70 :
                    use_computed_stuff()
                    break
                else :
                    opacity_from_tables=np.random.uniform(low=0.95*Opacity[-1], high=Opacity[-1])
            if iteration==15 and len(Opacity)>0 :
                if fmax>0.7 :
                    opacity_from_tables=Opacity[-1]
                else :
                    opacity_from_tables=(abs(opacity_temp+opacity_from_tables))/2
            if iteration==30 and len(Opacity)>0 :
                if fmax>0.7 :
                    opacity_from_tables=Opacity[-1]*1.5
                else :
                    opacity_from_tables=(abs(opacity_temp+opacity_from_tables))/2
            if iteration==60 :
                if len(Opacity)>0 :
                    opacity_from_tables=np.random.uniform(low=0.2*Opacity[-1], high=5.*Opacity[-1])
                    iteration=31
                    iteration_2=iteration_2+1
                    continue
                else :
                    opacity_from_tables=np.random.uniform(low=0.3/np.sqrt(float(iteration_2+1)), high=10.*np.sqrt(float(iteration_2+1)))
                    iteration=1
                    iteration_2=iteration_2+1
                    continue
            if iteration!=60 :
                iteration=iteration+1
                continue

    #------------------Interpolate profiles
    #The opacity iteration process above "fails" for high spin and high mdot, i.e. for some radii f(r) and the "right" opacity are not found
    #Here I try to correct the radial profiles for any numerical "error"
    #if problematic_radii has too many consecutive radii, then the cubic interpolation does something funny.
    #if it's the case, increase jmax (the log-step of the radii array) or, if this does not change much, use a more brutal linear interpolation for that object
    if len(f_array)!=len(set(f_array)) :
        problematic_radii=[]
        radius_array_norepet=[]
        f_array_norepet=[]
        opacity_norepet=[]
        for index in range(1,len(f_array)) :
            if f_array[index]==f_array[index-1] :
                problematic_radii.append(r[1:][index])
            else :
                radius_array_norepet.append(r[1:][index])
                f_array_norepet.append(f_array[index])
                opacity_norepet.append(Opacity[index])
        f_interpolation = interpolate.interp1d(radius_array_norepet, f_array_norepet, kind='cubic')
        opacity_interpolation = interpolate.interp1d(radius_array_norepet, opacity_norepet, kind='cubic')
        for i_radius in range (1, len(r)):
            if r[i_radius] in problematic_radii :
                new_f=f_interpolation(r[i_radius])
                new_opacity=opacity_interpolation(r[i_radius])
                f_array[i_radius]=new_f
                Opacity[i_radius]=new_opacity
                Q[i_radius]=Q_mmdot_terms*Q_radial_terms[i_radius]
                Qplus[i_radius]=Q_mmdot_terms*Q_radial_terms[i_radius]*2*np.pi*r[i_radius]*(r[i_radius]-r[i_radius-1])*(Rs_mmdot**2)
                if r[i_radius] in radius_rad :
                    right_index=np.where(radius_rad==r[i_radius])[0][0]
                    sol_rad[right_index]=new_f
                    radius_rad[right_index]=r[i_radius]
                    Lx_rad[right_index]=func_Lxmono(new_f*(1-downward_scattering)*Q_mmdot_terms*Q_radial_terms[i_radius]*2*np.pi*r[i_radius]*(r[i_radius]-r[i_radius-1])*(Rs_mmdot**2))
                    Lx_broadband_rad[right_index]=func_Lxbroad(new_f*(1-downward_scattering)*Q_mmdot_terms*Q_radial_terms[i_radius]*2*np.pi*r[i_radius]*(r[i_radius]-r[i_radius-1])*(Rs_mmdot**2))
                    radius_Ldisk_rad[right_index]=r[i_radius]
                    Tem_rad[right_index]=T_nof_mmdot_terms*T_nof_radial_terms[i_radius]*T_nof_constant_terms*(1-new_f*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_rad_constants*Temperature_term_rad_mmdot_terms*Temperature_term_rad_radial[i_radius]
                    Tmid_rad[right_index]=T_nof_mmdot_terms*T_nof_radial_terms[i_radius]*T_nof_constant_terms*(new_opacity/kes)**((2*mu-1)/(mu+4))*(1-new_f*(1-downward_scattering*(1-albedo)))**((2*mu-1)/(mu+4))
                    Ldisk_rad[right_index]=2*np.pi*r[i_radius]*(r[i_radius]-r[i_radius-1])*(Rs_mmdot**2)*BB_Flux_mononu(T_nof_mmdot_terms*T_nof_radial_terms[i_radius]*T_nof_constant_terms*(1-new_f*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_rad_constants*Temperature_term_rad_mmdot_terms*Temperature_term_rad_radial[i_radius])
                    rho_rad[right_index]=rho_mmdot_terms_rad*rho_radial_terms_rad[i_radius]*rho_constant_terms_rad*(new_opacity/kes)**(6*(mu-2)/(mu+4))*(1-new_f*(1-downward_scattering*(1-albedo)))**(6*(mu-2)/(mu+4))
                    P_rad[right_index]=P_mmdot_terms_rad*P_radial_terms_rad[i_radius]*P_constant_terms_rad*(new_opacity/kes)**(4*(2*mu-1)/(mu+4))*(1-new_f*(1-downward_scattering*(1-albedo)))**(4*(2*mu-1)/(mu+4))
                    h_rad[right_index]=h_propconst*(new_opacity/kes)*mdot_input*J(r[i_radius])*(1-new_f*(1-downward_scattering*(1-albedo)))
                    Derivative_Opacity[right_index]=Derivative_Opacity_closest(Tmid_rad[i_radius], rho_rad[i_radius])[0]
                    A[right_index]=A_constant_terms*(new_opacity/kes)**(9./(mu+4))*A_mmdot_terms*A_radial_terms[i_radius]
                    T_nof[right_index]=T_nof_mmdot_terms*T_nof_radial_terms[i_radius]*T_nof_constant_terms*(new_opacity/kes)**((2*mu-1)/(mu+4))
                    rho_nof[right_index]=rho_mmdot_terms_rad*rho_radial_terms_rad[i_radius]*rho_constant_terms_rad*(new_opacity/kes)**(6*(mu-2)/(mu+4))
                    P_nof[right_index]=P_mmdot_terms_rad*P_radial_terms_rad[i_radius]*P_constant_terms_rad*(new_opacity/kes)**(4*(2*mu-1)/(mu+4))
                    h_nof[right_index]=h_propconst*(new_opacity/kes)*mdot_input*J(r[i_radius])
                elif r[i_radius] in radius_gas :
                    right_index=np.where(radius_gas==r[i_radius])[0][0]
                    sol_gas[right_index]=new_f
                    radius_gas[right_index]=r[i_radius]
                    radius_Ldisk_gas[right_index]=r[i_radius]
                    Lx_gas[right_index]=func_Lxmono(new_f*(1-downward_scattering)*Q_mmdot_terms*Q_radial_terms[i_radius]*2*np.pi*r[i_radius]*(r[i_radius]-r[i_radius-1])*(Rs_mmdot**2))
                    Lx_broadband_gas[right_index]=func_Lxbroad(new_f*(1-downward_scattering)*Q_mmdot_terms*Q_radial_terms[i_radius]*2*np.pi*r[i_radius]*(r[i_radius]-r[i_radius-1])*(Rs_mmdot**2))
                    Tem_gas[right_index]=T_gas_nof_mmdot_terms*T_gas_nof_radial_terms[i_radius]*T_gas_nof_constant_terms*(1-new_f*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_gas_constants*Temperature_term_gas_mmdot_terms*Temperature_term_gas_radial[i_radius]
                    Tmid_gas[right_index]=T_gas_nof_mmdot_terms*T_gas_nof_radial_terms[i_radius]*T_gas_nof_constant_terms*(new_opacity/kes)**(1./5)*(1-new_f*(1-downward_scattering*(1-albedo)))**(1./5)
                    Ldisk_gas[right_index]=2*np.pi*r[i_radius]*(r[i_radius]-r[i_radius-1])*(Rs_mmdot**2)*BB_Flux_mononu(T_gas_nof_mmdot_terms*T_gas_nof_radial_terms[i_radius]*T_gas_nof_constant_terms*(1-new_f*(1-downward_scattering*(1-albedo)))**(1./4)*Temperature_term_gas_constants*Temperature_term_gas_mmdot_terms*Temperature_term_gas_radial[i_radius])
                    P_gas[right_index]=P_mmdot_terms_gas*P_radial_terms_gas[i_radius]*P_constant_terms_gas*(new_opacity/kes)**(-1./10)*(1-new_f*(1-downward_scattering*(1-albedo)))**(-1./10)
                    h_gas[right_index]=h_mmdot_terms_gas*h_radial_terms_gas[i_radius]*h_constant_terms_gas*(new_opacity/kes)**(1./10)*(1-new_f*(1-downward_scattering*(1-albedo)))**(1./10)
                    rho_gas[right_index]=rho_mmdot_terms_gas*rho_radial_terms_gas[i_radius]*rho_constant_terms_gas*(new_opacity/kes)**(-3./10)*(1-new_f*(1-downward_scattering*(1-albedo)))**(-3./10)
                    A_gas[right_index]=A_gas_constant_terms*A_gas_mmdot_terms*A_gas_radial_terms[i_radius]*(new_opacity/kes)**(9./10)
                    T_gas_nof[right_index]=T_gas_nof_mmdot_terms*T_gas_nof_radial_terms[i_radius]*T_gas_nof_constant_terms*(new_opacity/kes)**(1./5)
                    rho_gas_nof[right_index]=rho_mmdot_terms_gas*rho_radial_terms_gas[i_radius]*rho_constant_terms_gas*(new_opacity/kes)**(-3./10)
                    P_gas_nof[right_index]=P_mmdot_terms_gas*P_radial_terms_gas[i_radius]*P_constant_terms_gas*(new_opacity/kes)**(-1./10)
                    h_gas_nof[right_index]=h_mmdot_terms_gas*h_radial_terms_gas[i_radius]*h_constant_terms_gas*(new_opacity/kes)**(1./10)

    #------------------Computing total luminosities
    #At each radius Lx and Ldisk computed were the luminosities in the annulus at a given radius
    #Their sum is already the integral!
    Ldisk_tot=np.sum(Ldisk_rad)+np.sum(Ldisk_gas)
    Ldisk_tot_nuFnu=nu_disk*Ldisk_tot
    Lcor_tot=np.sum(Lx_rad)+np.sum(Lx_gas)
    Lcor_tot_nuFnu=nu_x*Lcor_tot
    Lcor_broadband_tot=np.sum(Lx_broadband_rad)+np.sum(Lx_broadband_gas)

    #--------------------------------------------------------------
    #------------------Computing stabilities on the final solutions
    #Read Merloni (2003) for the stability conditions
    #--------------------Prad>> regime  instabilities
    def stability_condition_1(f_solution) :
        return ((mu+4)/(2*mu)+(mu+4)/(2*mu)*(-9./8.*(-((f_solution/(1-downward_scattering*(1-albedo)))**((2+mu)/mu)*k_1**(2./mu))+(2*alpha_0)**(1./mu)*f_solution*(((2*mu+8)/(9*mu))+(1-downward_scattering*(1-albedo))**(-1))-(2*alpha_0)**(1./mu)*((2*mu+8)/(9*mu)))/(f_solution*((2*alpha_0)**(1./mu)-(f_solution*k_1/(1-downward_scattering*(1-albedo)))**(2./mu)))+(2*mu-1.)/(2*mu))**(-1.))
    def stability_condition_2(f_solution) :
        return ((7*mu-4)/(mu+4)+(7*mu-8)/(mu+4)*(-9./8.*(-((f_solution/(1-downward_scattering*(1-albedo)))**((2+mu)/mu)*k_1**(2./mu))+(2*alpha_0)**(1./mu)*f_solution*(((2*mu+8)/(9*mu))+(1-downward_scattering*(1-albedo))**(-1))-(2*alpha_0)**(1./mu)*((2*mu+8)/(9*mu)))/(f_solution*((2*alpha_0)**(1./mu)-(f_solution*k_1/(1-downward_scattering*(1-albedo)))**(2./mu))))**(-1.))

    sol_final_rad=[]
    sol_final_rad_inst=[]
    sol_final_rad_adv=[]
    radius_solfinal_rad=[]
    radius_solfinal_rad_inst=[]
    radius_solfinal_rad_adv=[]
    for i in range(0, len(sol_rad)) :
        if mu==0.0 :
            adv_thresh=0.25*radius_rad[i]
            if ((h_propconst*(float(Opacity[np.where((radius_rad+radius_gas) == radius_rad[i])[0][0]])/kes)*mdot_input*J(radius_rad[i])*(1-sol_rad[i]*(1-downward_scattering*(1-albedo))))>radius_rad[i]-adv_thresh) :
                sol_final_rad_adv.append(sol_rad[i])
                radius_solfinal_rad_adv.append(radius_rad[i])
            else :
                #only thermal stability is checked, as it is dominant at faster timescales
                if 6.<4.- Derivative_Opacity[np.where((radius_rad+radius_gas) == radius_rad[i])[0][0]] :
                    sol_final_rad.append(sol_rad[i])
                    radius_solfinal_rad.append(radius_rad[i])
                else :
                    sol_final_rad_inst.append(sol_rad[i])
                    radius_solfinal_rad_inst.append(radius_rad[i])
        if mu!=0.0 :
            adv_thresh=0.25*radius_rad[i]
            if ((h_propconst*(float(Opacity[np.where((radius_rad+radius_gas) == radius_rad[i])[0][0]])/kes)*mdot_input*J(radius_rad[i])*(1-sol_rad[i]*(1-downward_scattering*(1-albedo))))>radius_rad[i]-adv_thresh) : #and ((9.14*m_dot[identif_rad[i]-1]*J(radius_rad[i])*(1-sol_rad[i]*(1-downward_scattering*(1-albedo))))<radius_rad[i]+adv_thresh) : I noticed that h>r-thresh is sufficient, solutions are surely up to h<r, can check adding the <r condition, they stay the same
                sol_final_rad_adv.append(sol_rad[i])
                radius_solfinal_rad_adv.append(radius_rad[i])
            else :
                #only thermal stability is considered right now
                if stability_condition_1(sol_rad[i])<4.- Derivative_Opacity[np.where((radius_rad+radius_gas) == radius_rad[i])[0][0]] : # and stability_condition_2(sol_rad[i]) >0. :
                    sol_final_rad.append(sol_rad[i])
                    radius_solfinal_rad.append(radius_rad[i])
                else :
                    sol_final_rad_inst.append(sol_rad[i])
                    radius_solfinal_rad_inst.append(radius_rad[i])

    #--------------------Pgas>> regime  instabilities
    sol_final_gas=[]
    sol_final_gas_inst=[]
    sol_final_gas_adv=[]
    radius_solfinal_gas=[]
    radius_solfinal_gas_inst=[]
    radius_solfinal_gas_adv=[]
    for i in range(0, len(sol_gas)) :
        if mu==0 :
            adv_thresh=0.75
            if (sol_gas[i]*(1-downward_scattering*(1-albedo))<1.-(alpha_0*mass_input*xi*k_0**(14./3)*(adv_thresh*radius_gas[i])**10)/((h_constgas*(float(Opacity[np.where((radius_rad+radius_gas) == radius_gas[i])[0][0]])/kes)**(1./10))**10*radius_gas[i]**(21./2)*mdot_input**2*J(radius_gas[i])**2) ) :
                sol_final_gas_adv.append(sol_gas[i])
                radius_solfinal_gas_adv.append(radius_gas[i])
            else :
                #only thermal stability, mu=0 is stable at Pgas>>
                if 5./2<4.- Derivative_Opacity[np.where((radius_rad+radius_gas) == radius_gas[i])[0][0]] :
                    sol_final_gas.append(sol_gas[i])
                    radius_solfinal_gas.append(radius_gas[i])
                else :
                    sol_final_gas_inst.append(sol_gas[i])
                    radius_solfinal_gas_inst.append(radius_gas[i])
        if mu!=0 :
            adv_thresh=0.75
            if (sol_gas[i]*(1-downward_scattering*(1-albedo))<1.-(alpha_0*mass_input*xi*k_0**(14./3)*(adv_thresh*radius_gas[i])**10)/((h_constgas*(float(Opacity[np.where((radius_rad+radius_gas) == radius_gas[i])[0][0]])/kes)**(1./10))**10*radius_gas[i]**(21./2)*mdot_input**2*J(radius_gas[i])**2) ) :
                sol_final_gas_adv.append(sol_gas[i])
                radius_solfinal_gas_adv.append(radius_gas[i])
            else :
                #only thermal stability is considered, anyway if stability is obtained, it will be also for viscous instab in Pgas>>
                if ((5./2+ 0.5*(2./5*((-9*(1-downward_scattering*(1-albedo)))/(8*sol_gas[i]*((2*alpha_0)**(1./mu)-(sol_gas[i]*k_1/(1-downward_scattering*(1-albedo)))**(2./mu)))*(-k_1**(2./mu)*(sol_gas[i]/(1-downward_scattering*(1-albedo)))**((2+mu)/mu)+((2*alpha_0)**(1./mu)*sol_gas[i]*(20./(9.*mu)+(1-downward_scattering*(1-albedo))**(-1.)))-(20./(9.*mu)*(2*alpha_0)**(1./mu))))+1./5)**(-1.))<4.- Derivative_Opacity[np.where((radius_rad+radius_gas) == radius_gas[i])[0][0]]) :
                    sol_final_gas.append(sol_gas[i])
                    radius_solfinal_gas.append(radius_gas[i])
                else :
                    sol_final_gas_inst.append(sol_gas[i])
                    radius_solfinal_gas_inst.append(radius_gas[i])
    #--------------------------------------------------------------

    #-----------------Computing fmean------------------------------
    weigth_for_fmean=np.array(Qplus)*2*np.pi*np.array(r[1:])
    fmean=np.average(np.array(f_array),weights=weigth_for_fmean)
    actual_fmean_continuum=fmean*(1-downward_scattering)

    #-----------------Saving data---------------------------------

    data_dir=name_data_dir
    datadir = os.path.join(data_dir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    def Save_data(arrays, names, file_name) :
        data={}
        for ii_thing, thing in enumerate(arrays) :
            thing=toiter(thing)
            try :
                thing=[float(i) for i in thing]
            except Exception :
                pass
            data.update({
                '%s' %(names[ii_thing]): thing
        })
        json.dump(data, open(file_name, 'w'), indent=4)

    """
    What are we saving here?

    For details check Arcodia et al. (2019) or send an email at arcodia@mpe.mpg.de.

    Disk and corona monochromatic luminosities are obtained at the energy in input (default is 3000A and 2keV).
    Units are erg cm-2 s-1 Hz-1.

    fmean, actualfmean: one is <f>, the other <f>(1-eta), i.e. the actual average fraction of accretion
                        power that we observe as primary continuum from the X-ray corona.
                        See Arcodia et al. (2019).

    f_array, radius_array: full array for f(r) and r, i.e. regardless of Prad>> or Pgas>> regions

    f_r, radius, Lcor, Lcor_broadband, Ldisk, Tem, Tmid, rho, P, h:
                        They are, respectively: fraction of power dissipated in the corona;
                        distance from the BH in units of Rs=2GM/c^2; Luminosity profile at the given energy
                        (default 2keV), each value is the luminosity given by the annulus at a given r_i-r_(i-1);
                        same but for the broadband L (default 2-10 keV); same for the disk emission (default 3000A);
                        effective surface temperature and midplane temperature (K); density (g cm-3); total Ptot=Prad+Pgas;
                        disk scale-height (units of Rs).
                        They all have a suffix "_rad" or "_gas", according to the Prad>>Pgas or
                        Pgas>>Prad regime at that radius (see Arcodia et al., 2019).

    Opacity, Qplus: opacity at the midplane; total accretion power per unit area.

    sol_final_*, sol_final_*_inst, sol_final_*_adv: arrays for f(r), divided in (thermally) stable and unstable
                                                    solutions, and possibly advection-contaminated solutions.
                                                    See Arcodia et al. (2019) and Merloni (2003) for details.
                                                    There are also the related radius_ arrays. The "*" is "rad"
                                                    or "gas" according to the Prad>>Pgas or Pgas>>Prad regime
                                                    at the given radius (see Arcodia et al., 2019).

    """
    stuff_to_save=[Ldisk_tot_nuFnu, Lcor_tot_nuFnu, Lcor_broadband_tot, fmean,
                   actual_fmean_continuum, f_array, r[1:], sol_rad,
                   radius_rad, Lx_rad, Lx_broadband_rad, Ldisk_rad,
                   Tem_rad, Tmid_rad, rho_rad, P_rad, h_rad, sol_gas,
                   radius_gas, Lx_gas, Lx_broadband_gas, Ldisk_gas,
                   Tem_gas, Tmid_gas, rho_gas, P_gas, h_gas, Opacity,
                   Qplus, sol_final_rad, sol_final_rad_inst,
                   sol_final_rad_adv, radius_solfinal_rad,
                   radius_solfinal_rad_inst, radius_solfinal_rad_adv,
                   sol_final_gas, sol_final_gas_inst,
                   sol_final_gas_adv, radius_solfinal_gas,
                   radius_solfinal_gas_inst, radius_solfinal_gas_adv]

    name_stuff_to_save=['Disk_Luminosity', 'Corona_Luminosity', 'Corona_broadband_Luminosity',
                        'fmean', 'actualfmean', 'f_array', 'radius_array', 'f_r_rad', 'radius_rad', 'Lcor_rad',
                       'Lcor_broadband_rad', 'Ldisk_rad', 'Tem_rad', 'Tmid_rad', 'rho_rad', 'Ptot_rad',
                        'h_rad', 'f_r_gas', 'radius_gas', 'Lcor_gas', 'Lcor_broadband_gas', 'Ldisk_gas',
                        'Tem_gas', 'Tmid_gas', 'rho_gas', 'Ptot_gas', 'h_gas', 'opacity', 'Accretion_power',
                        'stable_f_r_rad', 'unstable_f_r_rad', 'adv_f_r_rad', 'radius_stable_rad',
                        'radius_unstable_rad', 'radius_adv_rad', 'stable_f_r_gas', 'unstable_f_r_gas',
                        'adv_f_r_gas', 'radius_stable_gas', 'radius_unstable_gas', 'radius_adv_gas']

    Save_data(stuff_to_save,name_stuff_to_save,data_filename)
