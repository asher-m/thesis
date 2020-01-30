#!/usr/bin/env python

"""Create figure 3 of Vania's RAM/RBSP comparison paper"""

import datetime
import os.path

import numpy
import numpy.ma
import matplotlib

#matplotlib.use('ps')

import matplotlib.cm
import matplotlib.colors
import matplotlib.dates
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import spacepy.pycdf
import spacepy.toolbox
import spacepy.plot.utils

#revised version of rampy, should roll back into spacepy
import rampy_h_o

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['text.usetex']= True
matplotlib.rcParams['ps.usedistiller'] = 'xpdf'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 9

omni = True #omnidirection (else 45-degree pitch)

hope_pa_file = '../data/rbspb_ect-hope-PA-L3_20121101_v4.0.0.cdf'
hope_pa_file_prev = '../data/rbspb_ect-hope-PA-L3_20121031_v4.0.0.cdf'
mageis_pa_file = '../data/rbspb_rel01_ect-mageisHIGH-L3_20121101_v3.0.0.cdf'
mageis_pa_file_prev = '../data/rbspb_rel01_ect-mageisHIGH-L3_20121031_v3.0.0.cdf'
hope_omni_file = hope_pa_file
hope_omni_file_prev = hope_pa_file_prev
mageis_omni_file = mageis_pa_file
mageis_omni_file_prev = mageis_pa_file_prev

w05_trace = rampy_h_o.ramsat('../data/RBSPb_nei_W05.nc')
w05f_trace = rampy_h_o.ramsat('../data/RBSPb_nej_W05fix.nc')
vs_trace = rampy_h_o.ramsat('../data/RBSPb_neh_VS.nc')

if omni:
    w05_trace.create_omniflux()
    w05f_trace.create_omniflux()
    vs_trace.create_omniflux()

def get_h_45_pa(trace):
    """Extract the 45-degree flux from a trace"""
    idx = numpy.argmin(numpy.abs(numpy.rad2deg(numpy.arccos(
        trace.data['pa_grid'])) - 45))
    return trace.data['FluxH+'][:, idx, :]

def read_rbsp_omni():
    with spacepy.pycdf.CDF(hope_omni_file_prev) as f:
        hope_h_flux = f['FPDO'][...]
        hope_time = f['Epoch_Ion'][...]
        hope_energy = f['HOPE_ENERGY_Ion'][...]

    with spacepy.pycdf.CDF(hope_omni_file) as f:
        hope_h_flux = numpy.append(hope_h_flux, f['FPDO'][...], axis=0)
        hope_time =  numpy.append(hope_time, f['Epoch_Ion'][...], axis=0)
        hope_energy = numpy.append(hope_energy, f['HOPE_ENERGY_Ion'][...],
                                   axis=0)

    with spacepy.pycdf.CDF(mageis_omni_file_prev) as f:
        mageis_h_flux = f['FPDU_0to180'][...]
        mageis_time = f['Epoch_prot'][...]
        mageis_energy = f['FPDU_Energy'][...]
        #hope they're same across files!
        mageis_pa = f['FPDU_0to180_Alpha'][...]
        mageis_h_flux_min = f['FPDU_0to180'].attrs['VALIDMIN']
        mageis_h_flux_max = f['FPDU_0to180'].attrs['VALIDMAX']
        mageis_h_flux_fill = f['FPDU_0to180'].attrs['FILLVAL']

    with spacepy.pycdf.CDF(mageis_omni_file) as f:
        mageis_h_flux = numpy.append(mageis_h_flux, f['FPDU_0to180'][...], axis=0)
        mageis_time = numpy.append(mageis_time, f['Epoch_prot'][...], axis=0)
        mageis_energy = numpy.append(mageis_energy, f['FPDU_Energy'][...],
                                     axis=0)

    #now need to integrate flux across axis 1, weighted by sin of PA
    idx = (mageis_h_flux > mageis_h_flux_max) | \
           (mageis_h_flux < mageis_h_flux_min) | \
           (mageis_h_flux == mageis_h_flux_fill)
    weights = numpy.sin(numpy.deg2rad(mageis_pa))
    mageis_h_flux = numpy.ma.average(
        numpy.ma.masked_array(
        mageis_h_flux, mask=idx, fill_value=mageis_h_flux_fill),
        axis=1, weights=weights).filled(mageis_h_flux_fill)

    return filter_rbsp(mageis_time, mageis_energy, mageis_h_flux, hope_time,
           hope_energy, hope_h_flux)

def filter_rbsp(mageis_time, mageis_energy, mageis_h_flux, hope_time,
           hope_energy, hope_h_flux):
    #filter the mageis data to get rid of all-fill times
    idx = mageis_energy[:, 0] > 0
    mageis_h_flux = mageis_h_flux[idx, ...]
    mageis_time = mageis_time[idx, ...]
    mageis_energy = mageis_energy[idx, ...]
    if numpy.max(numpy.abs(numpy.diff(mageis_energy, axis=0))) > 0:
        raise RuntimeError('magEIS energy is time-dependent...')

    mageis_energy = mageis_energy[0, 0:16] #don't use high energy
    mageis_energy = spacepy.toolbox.bin_center_to_edges(mageis_energy)
    mageis_h_flux = mageis_h_flux[:, 0:16]
    mageis_h_flux = numpy.ma.masked_array(mageis_h_flux, mask=mageis_h_flux<0,
                                          fillval=-1e31)
    mageis_time = spacepy.toolbox.bin_center_to_edges(
        matplotlib.dates.date2num(mageis_time))

    #HOPE for this time is not doing switching, thank goodness
    if numpy.max(numpy.abs(numpy.diff(hope_energy, axis=0))) > 0:
        raise RuntimeError('HOPE energy is time-dependent...')

    hope_energy = hope_energy[0, 33:] / 1000.0 #throw away low energy; to keV
    hope_energy = spacepy.toolbox.bin_center_to_edges(hope_energy)
    hope_h_flux = hope_h_flux[:, 33:]
    hope_h_flux = numpy.ma.masked_array(hope_h_flux, mask=hope_h_flux<0,
                                        fillval=-1e31)
    hope_time = spacepy.toolbox.bin_center_to_edges(
        matplotlib.dates.date2num(hope_time))
    return mageis_time, mageis_energy, mageis_h_flux, hope_time, \
           hope_energy, hope_h_flux

def read_rbsp_45():
    with spacepy.pycdf.CDF(hope_pa_file_prev) as f:
        hope_h_flux = f['FPDU'][:, 3, :] #This is pitch angle 54
        hope_time = f['Epoch_Ion'][...]
        hope_energy = f['HOPE_ENERGY_Ion'][...]
        
    with spacepy.pycdf.CDF(hope_pa_file) as f:
        hope_h_flux = numpy.append(hope_h_flux, f['FPDU'][:, 3, :], axis=0)
        hope_time =  numpy.append(hope_time, f['Epoch_Ion'][...], axis=0)
        hope_energy = numpy.append(hope_energy, f['HOPE_ENERGY_Ion'][...],
                                   axis=0)
        
    with spacepy.pycdf.CDF(mageis_pa_file_prev) as f:
        mageis_h_flux = f['FPDU'][:, 3, :] #pitch angle 48
        mageis_time = f['Epoch_prot'][...]
        mageis_energy = f['FPDU_Energy'][...]

    with spacepy.pycdf.CDF(mageis_pa_file) as f:
        mageis_h_flux = numpy.append(mageis_h_flux, f['FPDU'][:, 3, :], axis=0)
        mageis_time = numpy.append(mageis_time, f['Epoch_prot'][...], axis=0)
        mageis_energy = numpy.append(mageis_energy, f['FPDU_Energy'][...],
                                     axis=0)

    return filter_rbsp(mageis_time, mageis_energy, mageis_h_flux, hope_time,
           hope_energy, hope_h_flux)

mageis_time, mageis_energy, mageis_h_flux, \
             hope_time, hope_energy, hope_h_flux = \
             (read_rbsp_omni() if omni else read_rbsp_45())

if omni:
    w05_flux = w05_trace.omniH
    w05f_flux = w05f_trace.omniH
    vs_flux = vs_trace.omniH
else:
    w05_flux = get_h_45_pa(w05_trace)
    w05f_flux = get_h_45_pa(w05f_trace)
    vs_flux = get_h_45_pa(vs_trace)

if not (w05_trace.time == vs_trace.time).all():
    raise RuntimeError('Not on the same timebase!')

if not (w05_trace.data['energy_grid'] == vs_trace.data['energy_grid']).all():
    raise RuntimeError('Not on the same energy grid!')

timelim = [datetime.datetime(2012, 11, 1, 0), datetime.datetime(2012, 11, 2, 0)]
timelab = 'Universal Time from %s' % timelim[0].isoformat()
energlab= 'Energy (keV)'
fluxlab = 'flux (cm$^{-2}$s$^{-1}$sr$^{-1}$keV$^{-1}$)'
time = matplotlib.dates.date2num(w05_trace.time)
egrid= w05_trace.data['energy_grid']
yrng = [0.1, max(egrid)]
Mtick, mtick, fmt = rampy_h_o.smart_timeticks(timelim)

maxFlux = 1.0e6
minFlux = 1.0
mapname = 'jet'


def axes_setup(axes, do_energy_label=True):
    axes.set_xlim(timelim)
    axes.set_ylim([1, 350])
    axes.set_yscale('log')
    axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 3, 10, 30, 100, 350]))
    #there should be a better way to do this, but it makes things work.
    axes.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator([1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                                                 100, 200, 300]))
    if do_energy_label:
        axes.set_ylabel(energlab)
    axes.xaxis.set_major_locator(Mtick)
    axes.xaxis.set_minor_locator(mtick)

def do_plot(axes, time, energy, fluxes):
    return axes.pcolormesh(time, energy, fluxes,
                           norm=matplotlib.colors.LogNorm(), vmin=minFlux,
                           vmax=maxFlux, cmap=colormap,
                           shading='flat', edgecolors='None', rasterized=True)

def do_labels(axes, leftlabel, rightlabel):
    axes.text(-0.115, 0.5, leftlabel, transform=axes.transAxes, size=10)
    axes.text(1.0095, 0.5, rightlabel, transform=axes.transAxes)

#second dim is number of rows * 2, where hope+mageis H+ combined
#a standard H+/O+ RAM output is 2
#HOPE+magEIS combine to same (1 to 350)
#HOPE alone is 1 to 60, magEIS 60 to 350. In log space,
#HOPE is 1.4 (70% of 1 to 350 in log space)
#magEIS is 0.6 (30% of 1 to 350 in log space)
#Just 4 full plots: mageis+HOPE, 2 W05, 1 VS
height = 8.0
fig = plt.figure(figsize=(8, height),
                 subplotpars=
                 matplotlib.figure.SubplotParams(top=0.89, hspace=0.05),
                 dpi=300)
gs = matplotlib.gridspec.GridSpec(
    5, 1, height_ratios=[0.3, 0.7, 1, 1, 1])
colormap = matplotlib.cm.get_cmap(mapname)
colormap.set_bad(color='w')

fxmagEIS = fig.add_subplot(gs[0])
flxmagEIS = do_plot(fxmagEIS, mageis_time,
                    mageis_energy, mageis_h_flux.transpose())
axes_setup(fxmagEIS, do_energy_label=False)
fxmagEIS.set_ylim([60, 350])
plt.setp(fxmagEIS.get_xticklabels(), visible=False) #no X labels
do_labels(fxmagEIS, '(a)', 'magEIS')

fxHOPE = fig.add_subplot(gs[1], sharex=fxmagEIS)
flxHOPE = do_plot(fxHOPE, hope_time,
                    hope_energy, hope_h_flux.transpose())
axes_setup(fxHOPE, do_energy_label=False)
spacepy.plot.utils.shared_ylabel([fxHOPE, fxmagEIS], energlab)
fxHOPE.set_ylim([1, 60])
plt.setp(fxHOPE.get_xticklabels(), visible=False) #no X labels
do_labels(fxHOPE, '(b)', 'HOPE')

gs_idx = 2
fxW05 = fig.add_subplot(gs[gs_idx], sharex=fxmagEIS)
flxW05 = do_plot(fxW05, time, egrid, w05_flux[:-1,:-1].transpose())
axes_setup(fxW05)
plt.setp(fxW05.get_xticklabels(), visible=False) #no X labels
do_labels(fxW05, '(c)', 'W05')

gs_idx = 3
fxW05f = fig.add_subplot(gs[gs_idx], sharex=fxmagEIS)
flxW05f = do_plot(fxW05f, time, egrid, w05f_flux[:-1,:-1].transpose())
axes_setup(fxW05f)
plt.setp(fxW05f.get_xticklabels(), visible=False) #no X labels
do_labels(fxW05f, '(d)', 'W05\nfix b.c.')

gs_idx = 4
fxVS = fig.add_subplot(gs[gs_idx], sharex=fxW05) #same X axis!
flxVS = do_plot(fxVS, time, egrid, vs_flux[:-1,:-1].transpose())
axes_setup(fxVS)
do_labels(fxVS, '(e)', 'V-S')

x_axes_on = fxVS
newfmt = matplotlib.ticker.FuncFormatter(vs_trace.orbit_formatter)
x_axes_on.xaxis.set_major_formatter(newfmt)
x_axes_on.set_xlabel(timelab)
x_axes_on.set_xlabel('Universal Time from '+timelim[0].strftime('%Y-%m-%d'))
spacepy.plot.utils.annotate_xaxis(
    '\ $\mathrm{UT}$\n\ $\mathrm{MLT}$\n\ $\mathrm{R_E}$', x_axes_on)
axes = [fxmagEIS, fxW05, fxW05f, fxVS, fxHOPE]
cbar = plt.colorbar(flxW05, ax=axes,
                    pad=0.09, fraction=0.07, #extend='both', extendfrac='auto',
                    aspect=40, #same width for 4 plots as fig 2 for 2 plots
                    shrink=0.95, #slight shrink to give space top/bottom
                    ticks=matplotlib.ticker.LogLocator(), 
                    format=matplotlib.ticker.LogFormatterMathtext())
cbar.set_label(fluxlab)

#Combine HOPE and magEIS H+ plots
keep = [fxW05, fxW05f, fxVS]
lw = fxmagEIS.spines['top'].get_lw()
#add a line at the HOPE/magEIS break
for ax in keep:
    ax.axhline(60, color='k', lw=lw)
spacepy.plot.utils.collapse_vertical([fxmagEIS, fxHOPE], keep)
#turn back on the axis line only between HOPE and magEIS
fxmagEIS.spines['bottom'].set_visible(True)
fxHOPE.spines['top'].set_visible(True)

name = os.path.join('..', 'plots', 'figure_3' + ('_omni' if omni else '_45'))
fig.savefig(name + '.eps', dpi=300, bbox_inches='tight', pad_inches=0.0)
fig.savefig(name + '.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
