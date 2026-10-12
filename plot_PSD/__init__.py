"""
Power spectral density plotter with ACLR annotation, and plot_SEM.
"""
import numpy as np
import matplotlib.pyplot as plt

from nr_signal_utils import spectrum

def plot_PSD(**kwargs):

        """ Method for plotting PSD of one signal. If BW is given and no_plot is 0, will add ACLR levels to figure.

            Parameters
            ----------
            signal : array 
                Signal to be plotted
            Fc : integer
                Carrier frequency
            Fs : integer
                Sampling frequency
            BW : integer
                Bandwidth of whole signal
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
            self.plot_PSD(signal=[1+....-0.5],Fc=1e9,Fs=10e9, BW=200e6, double_sided = True, label= 'string' )

        """
        s=[] #self.rf_signal

        ACLR_offset=kwargs.get('ACLR_offset',0)

        BW=kwargs.get('BW',0)
        BW_conf=kwargs.get('BW_conf',0)
        ACLR_BW=kwargs.get('ACLR_BW',0)
        Fc=kwargs.get('Fc',0)

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
        if hasattr(BW,"__len__") :
            f_span=kwargs.get('f_span', BW+100E6)
            if hasattr(f_span,"__len__") :
                BW_tot=sum(abs(BW))
                f_span=BW_tot+4*max(BW)
        else:
            f_span=kwargs.get('f_span', BW+100E6)

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

        if hasattr(BW,"__len__") and len(BW)>1 :
            if not hasattr(BW_conf,"__len__"):
                BW_conf=[BW_conf]
            if not hasattr(ACLR_BW,"__len__"):
                ACLR_BW=[ACLR_BW]
            ACLR=[]
            matrix=np.transpose(np.vstack((f,Pxx)))
            matrix=matrix[matrix[:,0].argsort()]
            BW_vect_abs=abs(BW)
            f_off=[]
            l=[]
            h=[]
            chpow=[]
            BW_idx=0
            for i in range(0,len(BW)):
                BWi=BW[i] 
                if BWi>0:
                    offset=np.sum(BW_vect_abs[0:i])+BWi/2-BW_tot/2
                    f_off.append(offset)
                    lf=(np.abs(matrix[:,0]-(Fc+offset-0.5*BW_conf[BW_idx]))).argmin()
                    hf=(np.abs(matrix[:,0]-(Fc+offset+0.5*BW_conf[BW_idx]))).argmin()

                    if i==0:
                        min_f=min(f)
                    elif BW[i-1]<0 and abs(BW[i-1])>=BWi:
                        min_f=hf
                    else:
                        min_f=hf+abs(BW[i-1])
                    chpow=np.mean(matrix[lf:hf,1])
                    lf=matrix[lf,0]
                    hf=matrix[hf,0]
                    carriers_left=(lf-min_f)/BWi
                    if carriers_left>=2 and min_f<=lf-2.5*BWi:
                        carriers_left=2
                    elif carriers_left>=1 and  min_f<=lf-1.5*BWi:
                        carriers_left=1
                    else:
                        carriers_left=0
                    if i < len(BW)-1:
                        if BW[i+1]<0 and abs(BW[i+1])>=BWi:
                            max_f=hf+abs(BW[i+1])

                        else:
                            max_f=hf

                    else:
                        max_f=max(f)
                    carriers_right=(max_f-hf)/BWi
                    if carriers_right>=2 and max_f>=hf-2.5*BWi:
                        carriers_right=2
                    elif carriers_right>=1 and  max_f>=hf-1.5*BWi:
                        carriers_right=1
                    else:
                        carriers_right=0

                    for i in range(-int(carriers_left),int(carriers_right+1)):
                        if i ==0:
                            l=(np.abs(matrix[:,0]-(Fc+offset+(i-0.5)*BW_conf[BW_idx]))).argmin()
                            h=(np.abs(matrix[:,0]-(Fc+offset+(i+0.5)*BW_conf[BW_idx]))).argmin()
                        else:    
                            if ACLR_offset==0:
                                l=(np.abs(matrix[:,0]-(Fc+offset+(i*BWi-0.5*ACLR_BW[BW_idx])))).argmin()
                                h=(np.abs(matrix[:,0]-(Fc+offset+(i*BWi+0.5*ACLR_BW[BW_idx])))).argmin()
                            else: 
                                l=(np.abs(matrix[:,0]-(Fc+ACLR_offset+(i*BWi-0.5*ACLR_BW[BW_idx])))).argmin()
                                h=(np.abs(matrix[:,0]-(Fc+ACLR_offset+(i*BWi+0.5*ACLR_BW[BW_idx])))).argmin()
                        ac=np.mean(matrix[l:h,1])/chpow
                        ac_plot=np.mean(matrix[l:h,1])/max(matrix[:,1])
                        ACLR.append(10*np.log10(ac))
                        if no_plot==0:
                            if i==0:
                                pass
                            else:
                                ax.hlines(10*np.log10(ac_plot),(Fc+offset+(i*BWi-0.5*ACLR_BW[BW_idx]))/(freq_scale),(Fc+offset+(i*BWi+0.5*ACLR_BW[BW_idx]))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                ax.text(x=(Fc+offset+(i-0.5)*BWi)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')

                    BW_idx+=1

        else:

            if BW!=0 :
                if BW_conf==0:
                    BW_conf=BW
                if ACLR_BW==0:
                    ACLR_BW=BW_conf
                ACLR=[]
                matrix=np.transpose(np.vstack((f,Pxx)))
                matrix=matrix[matrix[:,0].argsort()]
                l=(np.abs(matrix[:,0]-(Fc-0.5*BW_conf))).argmin()
                h=(np.abs(matrix[:,0]-(Fc+0.5*BW_conf))).argmin()
                chpow=np.mean(matrix[l:h,1])

                ncarrier=(max(f)-min(f))/BW
                if ncarrier>=5 and max(f)>Fc+2.5*BW and min(f)<Fc-2.5*BW:
                    ncarrier=5
                elif ncarrier>=3 and max(f)>=Fc+1.5*BW and min(f)<=Fc-1.5*BW:
                    ncarrier=3
                else:
                    ncarrier=0

                for i in range(-int(np.floor(ncarrier/2)),int(np.floor(ncarrier/2)+1)):
                    if i ==0:
                        l=(np.abs(matrix[:,0]-(Fc+(i-0.5)*BW_conf))).argmin()
                        h=(np.abs(matrix[:,0]-(Fc+(i+0.5)*BW_conf))).argmin()
                    else:    
                        if ACLR_offset==0:
                            l=(np.abs(matrix[:,0]-(Fc+(i*BW-0.5*ACLR_BW)))).argmin()
                            h=(np.abs(matrix[:,0]-(Fc+(i*BW+0.5*ACLR_BW)))).argmin()
                        else:
                            l=(np.abs(matrix[:,0]-(Fc+(i*ACLR_offset-0.5*ACLR_BW)))).argmin()
                            h=(np.abs(matrix[:,0]-(Fc+(i*ACLR_offset+0.5*ACLR_BW)))).argmin()
                    ac=np.mean(matrix[l:h,1])/chpow
                    ACLR.append(10*np.log10(ac))
                    ac_plot=np.mean(matrix[l:h,1])/max(matrix[:,1])
                    if no_plot==0:
                        if i==0:
                            pass
                        else:
                            if ACLR_offset==0:
                                ax.hlines(10*np.log10(ac_plot),(Fc+(i*BW-0.5*ACLR_BW))/(freq_scale),(Fc+(i*BW+0.5*ACLR_BW))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                ax.text(x=(Fc+(i-0.5)*BW)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')
                            else:
                                ax.hlines(10*np.log10(ac_plot),(Fc+(i*ACLR_offset-0.5*ACLR_BW))/(freq_scale),(Fc+(i*ACLR_offset+0.5*ACLR_BW))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                ax.text(x=(Fc+(i-0.5)*ACLR_offset)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')

        if no_plot==0:
            if zoom_plt == 1:
                ax.set_xlim(f_plot[f_ctr_min_idx]/(freq_scale),f_plot[f_ctr_max_idx]/(freq_scale))
            if not isinstance(PSD_min,str):
                plt.ylim(PSD_min,10)
            plt.show(block=False)
        if no_plot==0:
            if hasattr(BW,"__len__") and len(BW)>1:
                return fig, ACLR

            if BW!=0:
                return fig, ACLR 
            return fig
        else:
            if hasattr(BW,"__len__") and len(BW)>1:
                return f_plot/(freq_scale),y_plot, ACLR
            if BW!=0:
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
