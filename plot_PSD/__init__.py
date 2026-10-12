"""
Power spectral density plotter with ACLR annotation, and plot_SEM.
"""
import dataclasses

import numpy as np
import matplotlib.pyplot as plt

from nr_signal_utils import measurements
from nr_signal_utils import plots
from nr_signal_utils import spectrum

def plot_PSD(**kwargs):

        """ Method for plotting PSD of one signal. If carriers is given and no_plot is 0, will add ACLR levels to figure.

            Parameters
            ----------
            signal : array 
                Signal to be plotted
            Fs : integer
                Sampling frequency
            carriers : list of nr_signal_utils.carrier.Carrier
                Carriers of the signal, for the ACLR levels
            no_plot : binary
                If 1, will return y and x- axis data (and ACLR) without plotting.
            double_sided: Boolean
                Double sided plot around central frequency

            legend: string
                Label for the plotted figure

            decim: int 
                decimation factor
            unit: string 
                Set to "kHz", "MHz", "GHz" - Default "GHz"

            PSD_min: int or str
                if int -> set the minimum value of PSD, 
                if str -> dynamically calculate min automatically
            Example
            -------
            plot_PSD(signal=[1+....-0.5],Fs=10e9, carriers=[Carrier("FR1", 100e6, 1, 0.5e-3, "64QAM")], double_sided = True, legend= 'string' )

        """
        s=[] #self.rf_signal

        ACLR_offset=kwargs.get('ACLR_offset',0)

        carriers=kwargs.get('carriers',[])
        Fc=0
        if len(carriers)==1:
            BW=carriers[0].bw
            Fc=carriers[0].offset

        Fs=int(kwargs.get('Fs',0))

        no_plot=kwargs.get('no_plot', 0)

        zoom_plt=kwargs.get('zoom_plt',0)
        decim=kwargs.get('decim',0)

        unit = kwargs.get('unit',"GHz")

        if unit.upper() == "GHZ":
            freq_scale  = 10**9
            unit_txt    = "GHz"
        elif unit.upper() == "MHZ":
            freq_scale  = 10**6 
            unit_txt    = "MHz"
        elif unit.upper() == "KHZ":
            freq_scale  = 10**3
            unit_txt    = "kHz"

        PSD_min = kwargs.get('PSD_min', -60)

        f_centre=kwargs.get('f_centre', Fc)
        if 'f_span' in kwargs:
            f_span=kwargs.get('f_span')
        elif len(carriers)>1:
            bws=[c.bw for c in carriers]
            f_span=sum(bws)+4*max(bws)
        elif len(carriers)==1:
            f_span=BW+100E6
        else:
            f_span=100E6

        if zoom_plt ==1:
            if decim==0:
                f_centre_max = f_centre + f_span/2
                f_centre_min = f_centre - f_span/2
            else:
                f_centre_max = 0 + f_span/2
                f_centre_min = 0 - f_span/2

        d_s = kwargs.get('double_sided', "False")
        if 'signal' in kwargs:
            s=kwargs.get('signal')

        legend = kwargs.get('legend', '')
        if s.real.all == 0:
            s = s.imag
        else:
            s = s.real

        if decim!=0 :
            stages=4
            t=np.arange(len(s))/Fs
            sign=s*np.exp(-1j*2*np.pi*f_centre*t)
            Fc=0
            f_centre=0
            der=np.array([1,-1])
            og_decim=decim

            for i in range(0,stages):
                sign=np.cumsum(sign)
                sign=sign[::int(np.log2(decim)/2)]
                sign=np.convolve(sign,der,'full')

            s=sign
            Fs=Fs/og_decim

        f,Pxx=spectrum.psd(s,Fs,onesided= not(d_s))

        y=10*np.log10(Pxx/max(Pxx))

        f_plot =f
        y_plot =y

        if zoom_plt == 1:

            f_ctr_max_diff = np.absolute(f_plot -  f_centre_max)
            f_ctr_max_idx  = f_ctr_max_diff.argmin()

            f_ctr_min_diff = np.absolute(f_plot -  f_centre_min)
            f_ctr_min_idx  = f_ctr_min_diff.argmin()

        if no_plot==0:
            ax=spectrum.plot_psd(f_plot,Pxx,unit=unit_txt,label=legend)
            fig=ax.figure

        if len(carriers)>0 :
            if decim!=0 and len(carriers)==1:
                carriers=[dataclasses.replace(carriers[0],offset=0)]
            result=measurements.aclr(f,Pxx,carriers,ACLR_offset)
            ACLR=[entry["aclr"] for entry in result]
            if no_plot==0:
                plots.annotate_aclr(ax,result,unit_txt)

        if no_plot==0:
            if zoom_plt == 1:
                ax.set_xlim(f_plot[f_ctr_min_idx]/(freq_scale),f_plot[f_ctr_max_idx]/(freq_scale))
            if not isinstance(PSD_min,str):
                plt.ylim(PSD_min,10)
            plt.show(block=False)
        if no_plot==0:
            if len(carriers)>0:
                return fig, ACLR
            return fig
        else:
            if len(carriers)>0:
                return f_plot/(freq_scale),y_plot, ACLR
            else:
                return f_plot/(freq_scale),y_plot

def plot_SEM(self,**kwargs):

        delta_f_obue=1500E6
        measuring_filter = 1E6

        operating_band_h=29.5E9
        operating_band_l=26.5E9

        f_offset_max = delta_f_obue
        delta_f_max  = f_offset_max-measuring_filter/2

        delta_f_l = 0.1*(operating_band_h-operating_band_l)
        delta_f_h = delta_f_max

        f_offset_l= 0.1*(operating_band_h-operating_band_l)+5E6
        f_offset_h= f_offset_max       

        if 'measuring_filter' in kwargs:
            measuring_filter=kwargs.get('measuring_filter')

        if 'delta_f_obue' in kwargs:
            delta_f_obue=kwargs.get('delta_f_obue')

        if 'operating_band_h' in kwargs:
            operating_band_h=kwargs.get('operating_band_h')

        if 'operating_band_l' in kwargs:
            operating_band_l=int(kwargs.get('operating_band_l'))

        if 'f_offset_max' in kwargs:
            f_offset_max=kwargs.get('f_offset_max')

        if 'delta_f_max' in kwargs:
            delta_f_max=kwargs.get('delta_f_max')

        if 'delta_f_l' in kwargs:
            delta_f_l=kwargs.get('delta_f_l')

        if 'delta_f_h' in kwargs:
            delta_f_h=kwargs.get('delta_f_h')

        if 'f_offset_l' in kwargs:
            f_offset_l=kwargs.get('f_offset_l')

        if 'f_offset_h' in kwargs:
            f_offset_h=kwargs.get('f_offset_h')

        frequency =[operating_band_l-delta_f_obue-1E9-2*operating_band_h,
                    operating_band_l-delta_f_obue-1E9,
                    operating_band_l-f_offset_h,
                    operating_band_l-delta_f_h,
                    operating_band_l-f_offset_l, 
                    operating_band_l-delta_f_l,
                    operating_band_l, 
                    operating_band_h,
                    operating_band_h+delta_f_l,
                    operating_band_h+f_offset_l,
                    operating_band_h+delta_f_h,
                    operating_band_h+f_offset_h,
                    operating_band_h+delta_f_obue+1E9,
                    operating_band_h+delta_f_obue+1E9+2*operating_band_h]
        limit = [-13, -13, -13,-13,-5,-5,0,0,-5,-5,-13,-13, -13, -13, ]

        return(frequency, limit)
