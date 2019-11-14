import numpy as np
import json
from DiskCoronaSim import toiter
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
from matplotlib import rcParams
from collections import OrderedDict
import re

def use_my_rcParams_options() :
    rcParams['font.size'] = '12'
    rcParams['figure.figsize'] = '8.0,6.0'
    rcParams['mathtext.default']='rm'
    rcParams['figure.autolayout']='True'
    rcParams['xtick.labelsize']='large'

def load_data_specific(mass_input,  mdot_input, photon_index=1.9, spin_input=0) :
    name_data_dir='data'
    data_filename=name_data_dir+'/Data_logm%s_mdot%s_gamma_%s_spin_%s.json'  %( str(round(np.log10(mass_input),2)), str(mdot_input), str(photon_index), str(spin_input))
    data_filename_cut='logm%s_mdot%s_gamma_%s_spin_%s' %( str(round(np.log10(mass_input),2)), str(mdot_input), str(photon_index), str(spin_input) )
    try :
        json_file=json.load(open(data_filename, 'r'))
        print("    Data file named: {} was loaded".format(data_filename))
        return json_file, data_filename_cut
    except Exception :
        print("    Could not find the file named: {}".format(data_filename))
        return None

def get_min_max(myjsonlist, myquantity_string) :
    min_temp=[]
    max_temp=[]
    if isinstance(myjsonlist,dict):
        min_temp.append(min(myjsonlist[myquantity_string]))
        max_temp.append(max(myjsonlist[myquantity_string]))
    else:
        for i_obj, obj in enumerate(myjsonlist) :
            min_temp.append(min(obj[myquantity_string]))
            max_temp.append(max(obj[myquantity_string]))
    return min(min_temp), max(max_temp)

def colormap_given_array(array, log) :
    """
    Simply define some colormaps to be used in the plots with multiple objects.
    """
    cmap = plt.cm.viridis
    if log==True :
        norm = matplotlib.colors.LogNorm(vmin=min(10**np.array(array)), vmax=max(10**np.array(array)))
        colormapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colormapper.set_array(np.logspace(min(array), max(array), 1000))
    else :
        norm = matplotlib.colors.Normalize(vmin=min(array), vmax=max(array))
        colormapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colormapper.set_array(np.logspace(min(np.log10(array)), max(np.log10(array)), 1000))
    return colormapper

def plot_data(data_json_list, data_json_filenames,
              plot_diskL=True,
              plot_Tmid=True,
              plot_Teff=True,
              plot_rho=True,
              plot_Ptot=True,
              plot_h=True,
              plot_k=True
             ) :
    """
    Plotting script that get one single json dict or a list of dict,
    and their filenames just to save plot with a consistent name.
    """

    single_object=False
    if isinstance(data_json_list,dict):
        single_object=True
    else:
        mdot_array=[]
        for i_obj, obj in enumerate(data_json_list) :
            mdot_string='_'.join(re.findall('mdot(.*?)\_', data_json_filenames[i_obj]))
            mdot_array.append(float(mdot_string))
        mdot_array=np.array(mdot_array)
        color_mdot=colormap_given_array(mdot_array, log=False)

    #get boundaries of the radii array for the plots
    rmin, rmax = get_min_max(data_json_list, 'radius_array')
    fmin, fmax = get_min_max(data_json_list, 'f_array')
    if plot_diskL :
        try :
            Luvmin = min(get_min_max(data_json_list, 'Ldisk_gas')[0],get_min_max(data_json_list, 'Ldisk_rad')[0])
            Luvmax = max(get_min_max(data_json_list, 'Ldisk_gas')[1],get_min_max(data_json_list, 'Ldisk_rad')[1])
        except :
            Luvmin, Luvmax = get_min_max(data_json_list, 'Ldisk_rad')

    use_my_rcParams_options()
    #------------------plot f(r) and Lx(r) by default
    fig_f = plt.figure()
    ax_f = fig_f.add_subplot(111)
    ax_f.set_xscale('log')
    ax_f.set_yscale('log')
    ax_f.set_xlim(rmin,rmax)
    ax_f.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_f.yaxis.set_major_formatter(ticker.NullFormatter())
    if fmin<0.1 and fmin>0.01 :
        plt.yticks((0.1,0.5,1),['0.1','0.5','1'])
    elif fmin<0.01 :
        plt.yticks((0.01, 0.1,0.5,1),['0.01','0.1','0.5','1'])
    else :
        plt.yticks((0.5,1),['0.5','1'])
    ax_f.set_ylim(fmin-0.1*fmin, 1.0)
    ax_f.set_xlabel(r'$R/R_{s}$', fontsize = 22)
    ax_f.set_ylabel(r'$f$',rotation='horizontal', fontsize = 22)
    #f_asympt=np.sqrt(2*alpha_0/(k_1**2.))
    plt.axhline(y=fmax, color='black', linestyle='--', lw=1.)
    if single_object==False :
        if len(data_json_list)>10 :
            print("   Plots are gonna be a bit crowded, implement a computation of a median (16th, 84th) profile.")
        for i_obj, obj in enumerate(data_json_list) :
            color = color_mdot.to_rgba(mdot_array[i_obj])
            ax_f.plot(obj['radius_rad'], obj['f_r_rad'], '.', markersize=8, color=color, lw=0)
            ax_f.plot(obj['radius_gas'], obj['f_r_gas'], '.', markersize=8, color=color, lw=0)
        plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
    else :
        ax_f.plot(data_json_list['radius_rad'], data_json_list['f_r_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
        ax_f.plot(data_json_list['radius_gas'], data_json_list['f_r_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
        handles, labels = fig_f.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax_f.legend(by_label.values(), by_label.keys(), loc=4, prop={'size': 15})
    if single_object :
        fig_f.savefig('data/fprofile_%s.pdf' %(data_json_filenames), format='pdf')
        fig_f.savefig('data/fprofile_%s.png' %(data_json_filenames), format='png')
    else :
        fig_f.savefig('data/fprofile_multiple.pdf', format='pdf')
        fig_f.savefig('data/fprofile_multiple.png', format='png')

    fig_Lx = plt.figure()
    ax_Lx = fig_Lx.add_subplot(111)
    ax_Lx.set_xscale('log')
    ax_Lx.set_yscale('log')
    ax_Lx.set_xlim(rmin,rmax)
    ax_Lx.set_xlabel(r'$R/R_{s}$', fontsize = 22)
    ax_Lx.set_ylabel(r'$L_{\nu,X}\,\,[erg\,\,s^{-1}\,\,Hz^{-1}]$',rotation='vertical', fontsize = 22)
    if single_object==False :
        for i_obj, obj in enumerate(data_json_list) :
            color = color_mdot.to_rgba(mdot_array[i_obj])
            ax_Lx.plot(obj['radius_rad'], obj['Lcor_rad'], '.', markersize=8, color=color, lw=0)
            ax_Lx.plot(obj['radius_gas'], obj['Lcor_gas'], '.', markersize=8, color=color, lw=0)
        plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
    else :
        ax_Lx.plot(data_json_list['radius_rad'], data_json_list['Lcor_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
        ax_Lx.plot(data_json_list['radius_gas'], data_json_list['Lcor_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
        handles, labels = fig_Lx.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax_Lx.legend(by_label.values(), by_label.keys(), loc=1, prop={'size': 15})
    if single_object :
        fig_Lx.savefig('data/Lxprofile_%s.pdf' %(data_json_filenames), format='pdf')
        fig_Lx.savefig('data/Lxprofile_%s.png' %(data_json_filenames), format='png')
    else :
        fig_Lx.savefig('data/Lxprofile_multiple.pdf', format='pdf')
        fig_Lx.savefig('data/Lxprofile_multiple.png', format='png')

    if plot_diskL :
        fig_Luv = plt.figure()
        ax_Luv = fig_Luv.add_subplot(111)
        ax_Luv.set_xscale('log')
        ax_Luv.set_yscale('log')
        if Luvmin<1e20 and Luvmax>1e20:
            ax_Luv.set_ylim(1e20,Luvmax+0.5*Luvmax)
        ax_Luv.set_xlim(rmin,rmax)
        ax_Luv.set_xlabel(r'$R/R_{s}$', fontsize = 22)
        ax_Luv.set_ylabel(r'$L_{\nu,UV}\,\,[erg\,\,s^{-1}\,\,Hz^{-1}]$',rotation='vertical', fontsize = 22)
        if single_object==False :
            for i_obj, obj in enumerate(data_json_list) :
                color = color_mdot.to_rgba(mdot_array[i_obj])
                ax_Luv.plot(obj['radius_rad'], obj['Ldisk_rad'], '.', markersize=8, color=color, lw=0)
                ax_Luv.plot(obj['radius_gas'], obj['Ldisk_gas'], '.', markersize=8, color=color, lw=0)
            plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
        else :
            ax_Luv.plot(data_json_list['radius_rad'], data_json_list['Ldisk_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
            ax_Luv.plot(data_json_list['radius_gas'], data_json_list['Ldisk_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
            handles, labels = fig_Luv.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax_Luv.legend(by_label.values(), by_label.keys(), loc=1, prop={'size': 15})
        if single_object :
            fig_Luv.savefig('data/Luvprofile_%s.pdf' %(data_json_filenames), format='pdf')
            fig_Luv.savefig('data/Luvprofile_%s.png' %(data_json_filenames), format='png')
        else :
            fig_Luv.savefig('data/Luvprofile_multiple.pdf', format='pdf')
            fig_Luv.savefig('data/Luvprofile_multiple.png', format='png')

    if plot_Tmid :
        fig_Tmid = plt.figure()
        ax_Tmid = fig_Tmid.add_subplot(111)
        ax_Tmid.set_xscale('log')
        ax_Tmid.set_yscale('log')
        ax_Tmid.set_xlim(rmin,rmax)
        ax_Tmid.set_xlabel(r'$R/R_{s}$', fontsize = 22)
        ax_Tmid.set_ylabel(r'$T(r,0)\,\,[K]$',rotation='vertical', fontsize = 22)
        if single_object==False :
            for i_obj, obj in enumerate(data_json_list) :
                color = color_mdot.to_rgba(mdot_array[i_obj])
                ax_Tmid.plot(obj['radius_rad'], obj['Tmid_rad'], '.', markersize=8, color=color, lw=0)
                ax_Tmid.plot(obj['radius_gas'], obj['Tmid_gas'], '.', markersize=8, color=color, lw=0)
            plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
        else :
            ax_Tmid.plot(data_json_list['radius_rad'], data_json_list['Tmid_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
            ax_Tmid.plot(data_json_list['radius_gas'], data_json_list['Tmid_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
            handles, labels = fig_Tmid.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax_Tmid.legend(by_label.values(), by_label.keys(), loc=1, prop={'size': 15})
        if single_object :
            fig_Tmid.savefig('data/Tmidprofile_%s.pdf' %(data_json_filenames), format='pdf')
            fig_Tmid.savefig('data/Tmidprofile_%s.png' %(data_json_filenames), format='png')
        else :
            fig_Tmid.savefig('data/Tmidprofile_multiple.pdf', format='pdf')
            fig_Tmid.savefig('data/Tmidprofile_multiple.png', format='png')

    if plot_Teff :
        fig_Tem = plt.figure()
        ax_Tem = fig_Tem.add_subplot(111)
        ax_Tem.set_xscale('log')
        ax_Tem.set_yscale('log')
        ax_Tem.set_xlim(rmin,rmax)
        ax_Tem.set_xlabel(r'$R/R_{s}$', fontsize = 22)
        ax_Tem.set_ylabel(r'$T(r,h)\,\,[K]$',rotation='vertical', fontsize = 22)
        if single_object==False :
            for i_obj, obj in enumerate(data_json_list) :
                color = color_mdot.to_rgba(mdot_array[i_obj])
                ax_Tem.plot(obj['radius_rad'], obj['Tem_rad'], '.', markersize=8, color=color, lw=0)
                ax_Tem.plot(obj['radius_gas'], obj['Tem_gas'], '.', markersize=8, color=color, lw=0)
            plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
        else :
            ax_Tem.plot(data_json_list['radius_rad'], data_json_list['Tem_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
            ax_Tem.plot(data_json_list['radius_gas'], data_json_list['Tem_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
            handles, labels = fig_Tem.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax_Tem.legend(by_label.values(), by_label.keys(), loc=1, prop={'size': 15})
        if single_object :
            fig_Tem.savefig('data/Temprofile_%s.pdf' %(data_json_filenames), format='pdf')
            fig_Tem.savefig('data/Temprofile_%s.png' %(data_json_filenames), format='png')
        else :
            fig_Tem.savefig('data/Temprofile_multiple.pdf', format='pdf')
            fig_Tem.savefig('data/Temprofile_multiple.png', format='png')

    if plot_rho :
        fig_rho = plt.figure()
        ax_rho = fig_rho.add_subplot(111)
        ax_rho.set_xscale('log')
        ax_rho.set_yscale('log')
        ax_rho.set_xlim(rmin,rmax)
        ax_rho.set_xlabel(r'$R/R_{s}$', fontsize = 22)
        ax_rho.set_ylabel(r'$\rho(r)\,\,[g\,\,cm^{-3}]$',rotation='vertical', fontsize = 22)
        if single_object==False :
            for i_obj, obj in enumerate(data_json_list) :
                color = color_mdot.to_rgba(mdot_array[i_obj])
                ax_rho.plot(obj['radius_rad'], obj['rho_rad'], '.', markersize=8, color=color, lw=0)
                ax_rho.plot(obj['radius_gas'], obj['rho_gas'], '.', markersize=8, color=color, lw=0)
            plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
        else :
            ax_rho.plot(data_json_list['radius_rad'], data_json_list['rho_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
            ax_rho.plot(data_json_list['radius_gas'], data_json_list['rho_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
            handles, labels = fig_rho.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax_rho.legend(by_label.values(), by_label.keys(), loc=1, prop={'size': 15})
        if single_object :
            fig_rho.savefig('data/rhoprofile_%s.pdf' %(data_json_filenames), format='pdf')
            fig_rho.savefig('data/rhoprofile_%s.png' %(data_json_filenames), format='png')
        else :
            fig_rho.savefig('data/rhoprofile_multiple.pdf', format='pdf')
            fig_rho.savefig('data/rhoprofile_multiple.png', format='png')

    if plot_Ptot :
        fig_Ptot = plt.figure()
        ax_Ptot = fig_Ptot.add_subplot(111)
        ax_Ptot.set_xscale('log')
        ax_Ptot.set_yscale('log')
        ax_Ptot.set_xlim(rmin,rmax)
        ax_Ptot.set_xlabel(r'$R/R_{s}$', fontsize = 22)
        ax_Ptot.set_ylabel(r'$P_{tot}(r)\,\,[dyn]$',rotation='vertical', fontsize = 22)
        if single_object==False :
            for i_obj, obj in enumerate(data_json_list) :
                color = color_mdot.to_rgba(mdot_array[i_obj])
                ax_Ptot.plot(obj['radius_rad'], obj['Ptot_rad'], '.', markersize=8, color=color, lw=0)
                ax_Ptot.plot(obj['radius_gas'], obj['Ptot_gas'], '.', markersize=8, color=color, lw=0)
            plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
        else :
            ax_Ptot.plot(data_json_list['radius_rad'], data_json_list['Ptot_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
            ax_Ptot.plot(data_json_list['radius_gas'], data_json_list['Ptot_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
            handles, labels = fig_Ptot.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax_Ptot.legend(by_label.values(), by_label.keys(), loc=1, prop={'size': 15})
        if single_object :
            fig_Ptot.savefig('data/Ptotprofile_%s.pdf' %(data_json_filenames), format='pdf')
            fig_Ptot.savefig('data/Ptotprofile_%s.png' %(data_json_filenames), format='png')
        else :
            fig_Ptot.savefig('data/Ptotprofile_multiple.pdf', format='pdf')
            fig_Ptot.savefig('data/Ptotprofile_multiple.png', format='png')

    if plot_h :
        fig_h = plt.figure()
        ax_h = fig_h.add_subplot(111)
        ax_h.set_xscale('log')
        ax_h.set_yscale('log')
        ax_h.set_xlim(rmin,rmax)
        ax_h.set_xlabel(r'$R/R_{s}$', fontsize = 22)
        ax_h.set_ylabel(r'$H/R_{s}$',rotation='vertical', fontsize = 22)
        lin_h_r=np.linspace(rmin,rmax,100)
        ax_h.plot(lin_h_r,lin_h_r, color='black',linestyle='--', linewidth=1, zorder=0)
        if single_object==False :
            for i_obj, obj in enumerate(data_json_list) :
                color = color_mdot.to_rgba(mdot_array[i_obj])
                ax_h.plot(obj['radius_rad'], obj['h_rad'], '.', markersize=8, color=color, lw=0)
                ax_h.plot(obj['radius_gas'], obj['h_gas'], '.', markersize=8, color=color, lw=0)
            plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
        else :
            ax_h.plot(data_json_list['radius_rad'], data_json_list['h_rad'], '.', markersize=8, color='red', lw=0, label=r'$P_{rad}$')
            ax_h.plot(data_json_list['radius_gas'], data_json_list['h_gas'], '.', markersize=8, color='blue', lw=0, label=r'$P_{gas}$')
            handles, labels = fig_h.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax_h.legend(by_label.values(), by_label.keys(), loc=1, prop={'size': 15})
        if single_object :
            fig_h.savefig('data/hprofile_%s.pdf' %(data_json_filenames), format='pdf')
            fig_h.savefig('data/hprofile_%s.png' %(data_json_filenames), format='png')
        else :
            fig_h.savefig('data/hprofile_multiple.pdf', format='pdf')
            fig_h.savefig('data/hprofile_multiple.png', format='png')

    if plot_k :
        fig_opacity = plt.figure()
        ax_opacity = fig_opacity.add_subplot(111)
        ax_opacity.set_xscale('log')
        ax_opacity.set_yscale('log')
        ax_opacity.set_xlim(rmin,rmax)
        ax_opacity.set_xlabel(r'$R/R_{s}$', fontsize = 22)
        ax_opacity.set_ylabel(r'$k\,\,[cm^{2}\,\,g^{-1}]$',rotation='vertical', fontsize = 22)
        ax_opacity.axhline(y=0.34, color='grey', linestyle='--', lw=1.0)
        ax_opacity.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax_opacity.yaxis.set_major_formatter(ticker.NullFormatter())
        plt.yticks((0.3,0.5,1),['0.3','0.5','1'])
        if single_object==False :
            for i_obj, obj in enumerate(data_json_list) :
                color = color_mdot.to_rgba(mdot_array[i_obj])
                ax_opacity.plot(obj['radius_array'], obj['opacity'], '-', markersize=8, color=color, lw=3)
            plt.colorbar(mappable=color_mdot, label=r'$\dot{m}$')
        else :
            ax_opacity.plot(data_json_list['radius_array'], data_json_list['opacity'], '-', markersize=8, color='black', lw=3)

        if single_object :
            fig_opacity.savefig('data/opacityprofile_%s.pdf' %(data_json_filenames), format='pdf')
            fig_opacity.savefig('data/opacityprofile_%s.png' %(data_json_filenames), format='png')
        else :
            fig_opacity.savefig('data/opacityprofile_multiple.pdf', format='pdf')
            fig_opacity.savefig('data/opacityprofile_multiple.png', format='png')
