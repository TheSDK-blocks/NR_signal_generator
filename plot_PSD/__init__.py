"""
=========
plot_PSD
=========

My Entity model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is important.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import fft as sp_fft

import multiprocessing
import time
import random
import plot_format
plot_format.set_style('isscc')

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

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
        def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):

            # Created strided array of data segments
            if nperseg == 1 and noverlap == 0:
                result = x[..., np.newaxis]
            else:
                # https://stackoverflow.com/a/5568169
                step = nperseg - noverlap
                shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
                strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
                result = np.lib.stride_tricks.as_strided(x, shape=shape,strides=strides)
              

            # Detrend each data segment individually
            result = detrend_func(result)

            # Apply window by multiplication
            result = win * result

            # Perform the fft. Acts on last axis by default. Zero-pads automatically
            if sides == 'twosided':
                func = sp_fft.fft
            else:
                result = result.real
                func = sp_fft.rfft
                  

            result = func(result, n=nfft, workers = os.cpu_count())

            return result

        def _triage_segments(window, nperseg, input_length):
   
    # parse window; if array like, then set nperseg = win.shape
            if isinstance(window, str) or isinstance(window, tuple):
                # if nperseg not specified
                if nperseg is None:
                    nperseg = 256  # then change to default
                if nperseg > input_length:
                    warnings.warn('nperseg = {0:d} is greater than input length '
                          ' = {1:d}, using nperseg = {1:d}'
                          .format(nperseg, input_length))
                    nperseg = input_length
                win = get_window(window, nperseg)
            else:
                win = np.asarray(window)
                if len(win.shape) != 1:
                    raise ValueError('window must be 1-D')
                if input_length < win.shape[-1]:
                    raise ValueError('window is longer than input signal')
                if nperseg is None:
                    nperseg = win.shape[0]
                elif nperseg is not None:
                    if nperseg != win.shape[0]:
                        raise ValueError("value specified for nperseg is different"
                                 " from length of window")
            return win, nperseg


        def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,nfft=None, detrend='constant', return_onesided=True,scaling='density', axis=-1, mode='psd', boundary=None, padded=False):
            

            if mode not in ['psd', 'stft']:
                raise ValueError("Unknown value for mode %s, must be one of: "
                         "{'psd', 'stft'}" % mode)

            boundary_funcs = {#'even': even_ext,
                              #'odd': odd_ext,
                              #'constant': const_ext,
                              #'zeros': zero_ext,
                               None: None}

            if boundary not in boundary_funcs:
                raise ValueError("Unknown boundary option '{0}', must be one of: {1}".format(boundary, list(boundary_funcs.keys())))

            # If x and y are the same object we can save ourselves some computation.
            same_data = y is x

            if not same_data and mode != 'psd':
                raise ValueError("x and y must be equal if mode is 'stft'")

            axis = int(axis)

            # Ensure we have np.arrays, get outdtype
            x = np.asarray(x)
            if not same_data:
                y = np.asarray(y)
                outdtype = np.result_type(x, y, np.complex64)
            else:
                outdtype = np.result_type(x, np.complex64)

            if not same_data:
        # Check if we can broadcast the outer axes together
                xouter = list(x.shape)
                youter = list(y.shape)
                xouter.pop(axis)
                youter.pop(axis)
                try:
                    outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
                except ValueError as e:
                    raise ValueError('x and y cannot be broadcast together.') from e

            if same_data:
                if x.size == 0:
                    return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
            else:
                if x.size == 0 or y.size == 0:
                    outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
                    emptyout = np.rollaxis(np.empty(outshape), -1, axis)
                    return emptyout, emptyout, emptyout

            if x.ndim > 1:
                print('>1')
                if axis != -1:
                    print('-1')
                    x = np.rollaxis(x, axis, len(x.shape))
                    if not same_data and y.ndim > 1:
                        y = np.rollaxis(y, axis, len(y.shape))

            # Check if x and y are the same length, zero-pad if necessary
            if not same_data:
                if x.shape[-1] != y.shape[-1]:
                    if x.shape[-1] < y.shape[-1]:
                        pad_shape = list(x.shape)
                        pad_shape[-1] = y.shape[-1] - x.shape[-1]
                        x = np.concatenate((x, np.zeros(pad_shape)), -1)
                    else:
                        pad_shape = list(y.shape)
                        pad_shape[-1] = x.shape[-1] - y.shape[-1]
                        y = np.concatenate((y, np.zeros(pad_shape)), -1)

            if nperseg is not None:# if specified by user
                nperseg = int(nperseg)
                if nperseg < 1:
                    raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
            win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

            if nfft is None:
                nfft = nperseg
            elif nfft < nperseg:
                raise ValueError('nfft must be greater than or equal to nperseg.')
            else:
                nfft = int(nfft)

            if noverlap is None:
                noverlap = nperseg//2
            else:
                noverlap = int(noverlap)
            if noverlap >= nperseg:
                raise ValueError('noverlap must be less than nperseg.')
            nstep = nperseg - noverlap

            # Padding occurs after boundary extension, so that the extended signal ends
            # in zeros, instead of introducing an impulse at the end.
            # I.e. if x = [..., 3, 2]
            # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
            # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

            if boundary is not None:
                ext_func = boundary_funcs[boundary]
                x = ext_func(x, nperseg//2, axis=-1)
                if not same_data:
                    y = ext_func(y, nperseg//2, axis=-1)

            if padded:
            # Pad to integer number of windowed segments
            # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
                nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
                zeros_shape = list(x.shape[:-1]) + [nadd]
                x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
                if not same_data:
                    zeros_shape = list(y.shape[:-1]) + [nadd]
                    y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
            if not detrend:
                def detrend_func(d):
                    return d
            
            elif not hasattr(detrend, '__call__'):
                def detrend_func(d):
                    return signaltools.detrend(d, type=detrend, axis=-1)
            elif axis != -1:

            # Wrap this function so that it receives a shape that it could
            # reasonably expect to receive.
                def detrend_func(d):
                    d = np.rollaxis(d, -1, axis)
                    d = detrend(d)
                    return np.rollaxis(d, axis, len(d.shape))
            else:
                detrend_func = detrend

            if np.result_type(win, np.complex64) != outdtype:
                win = win.astype(outdtype)

            if scaling == 'density':
                scale = 1.0 / (fs * (win*win).sum())
            elif scaling == 'spectrum':
                scale = 1.0 / win.sum()**2
            else:
                raise ValueError('Unknown scaling: %r' % scaling)

            if mode == 'stft':
                scale = np.sqrt(scale)

            if return_onesided:
                if np.iscomplexobj(x):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to '
                          'return_onesided=False')
                else:
                    sides = 'onesided'
                    if not same_data:
                        if np.iscomplexobj(y):
                            sides = 'twosided'
                            warnings.warn('Input data is complex, switching to '
                                  'return_onesided=False')
            else:
                sides = 'twosided'

            if sides == 'twosided':
                freqs = sp_fft.fftfreq(nfft, 1/fs)
            elif sides == 'onesided':
                freqs = sp_fft.rfftfreq(nfft, 1/fs)

        # Perform the windowed FFTs
            result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)
            if not same_data:
        # All the same operations on the y data
                result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,sides)
                result = np.conjugate(result) * result_y
            elif mode == 'psd':
                result = np.conjugate(result) * result
            result *= scale
            if sides == 'onesided' and mode == 'psd':
                if nfft % 2:
                    result[..., 1:] *= 2
                else:
                # Last point is unpaired Nyquist freq point, don't double
                    result[..., 1:-1] *= 2

            time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,nperseg - noverlap)/float(fs)
            if boundary is not None:
                time -= (nperseg/2) / fs

            result = result.astype(outdtype)

    # All imaginary parts are zero anyways
            if same_data and mode != 'stft':
                result = result.real

        # Output is going to have new last axis for time/window index, so a
        # negative axis index shifts down one
            if axis < 0:
                axis -= 1

    # Roll frequency axis back to axis where the data came from
            result = np.rollaxis(result, -1, axis)

            return freqs, time, result

        def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density',axis=-1, average='mean'):

            freqs, Pxx = csd(x, x, fs=fs, window=window, nperseg=nperseg,
                             noverlap=noverlap, nfft=nfft, detrend=detrend,
                             return_onesided=return_onesided, scaling=scaling,
                             axis=axis, average=average)

            return freqs, Pxx.real


        def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,detrend='constant', return_onesided=True, scaling='density',axis=-1, average='mean'):
            
            freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,detrend, return_onesided, scaling, axis,mode='psd')

            # Average over windows.
            if len(Pxy.shape) >= 2 and Pxy.size > 0:
                if Pxy.shape[-1] > 1:
                    if average == 'median':
                        Pxy = np.median(Pxy, axis=-1) / _median_bias(Pxy.shape[-1])
                    elif average == 'mean':
                        Pxy = Pxy.mean(axis=-1)
                    else:
                        raise ValueError('average must be "median" or "mean", got %s'
                                         % (average,))
                else:
                    Pxy = np.reshape(Pxy, Pxy.shape[:-1])

            return freqs, Pxy
                

        BW=0
        Fc=0
        no_plot=0
        
        zoom_plt = 0

        a=100
        s=[] #self.rf_signal
        

        ACLR_offset=kwargs.get('ACLR_offset',0)
 
        BW=kwargs.get('BW',0)
        BW_conf=kwargs.get('BW_conf',0)
        ACLR_BW=kwargs.get('ACLR_BW',0)
        Fc=kwargs.get('Fc',0)

        Fs=int(kwargs.get('Fs',0))

        no_plot=kwargs.get('no_plot', 0)

        x=kwargs.get('signal', 0)

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
        #pdb.set_trace() 


        if zoom_plt ==1:
            if decim==0:
                f_centre_max = f_centre + f_span/2
                f_centre_min = f_centre - f_span/2
                print('fcentre',f_centre)
                print('f_centre_max', f_centre_max)
                print('f_centre_min', f_centre_min)
            else:
                f_centre_max = 0 + f_span/2
                f_centre_min = 0 - f_span/2
                print('fcentre',0)
                print('f_centre_max', f_centre_max)
                print('f_centre_min', f_centre_min)


        
        print('BW',BW)

        d_s = kwargs.get('double_sided', "False")
        if 'signal' in kwargs:
            s=kwargs.get('signal')
    
        legend = kwargs.get('legend', '')
        if s.real.all == 0:
            s = s.imag
        else:
            s = s.real


        if decim!=0 :
            #pdb.set_trace() 
            stages=4
            t=np.arange(len(s))/Fs
            sign=s*np.exp(-1j*2*np.pi*f_centre*t)
            Fc=0
            f_centre=0
            #cic=np.concatenate((np.ones(decim),np.zeros(decim)))
            #fil=cic
            der=np.array([1,-1])
            #der=np.zeros(factor+1)
            #der[0]=1
            #der[-1]=-1
            fil=der
            og_decim=decim
            #pdb.set_trace()
            #sign=np.cumsum(sign)
            #sign=sign/(max([max(abs(np.real(sign))),max(abs(np.imag(sign)))]))
            #sign=sign/len(sign)
            
            for i in range(0,stages):
                #fil=np.convolve(fil,cic,'full')
                sign=np.cumsum(sign)
                sign=sign[::int(np.log2(decim)/2)]
                #decim=decim/4
                #sign=sign/(max([max(abs(np.real(sign))),max(abs(np.imag(sign)))]))
                #sign=sign/len(sign)
                sign=np.convolve(sign,der,'full')
            #pdb.set_trace()
            



            #for i in range(0,stages-1):
            #    #fil=np.convolve(fil,cic,'full')
            #    sign=np.cumsum(sign)
            #    #sign=sign/(max([max(abs(np.real(sign))),max(abs(np.imag(sign)))]))
            #    sign=sign/len(sign)
            #    fil=np.convolve(fil,der,'full')
            #pdb.set_trace()
            #sign=sign[::decim]
            #
            #s_fil=np.convolve(sign,fil,'full')

            #s_fil=s_fil[::decim]
            #s=s_fil
            s=sign
            Fs=Fs/og_decim
        #s=x.NRfilter(Fs,s,BW,x.self.osr)
        #print('s',x)
        #print('s',x.shape)
        #print('s', x[2])
        #s= x[:,1]+1j*x[:,2]

        Lsegm_perc = 10
        Fs_given = 0
        plot_color = 'k'
        win_type = 'tukey'
        param = 1
        overlap_perc = 50

        Fs_given = 1
        
        fmin = -Fs/2
        fmax = Fs/2

        #Lsegm_perc = a
        #a=len(s)
        Lsegm = round(len(s)*Lsegm_perc/100)
        noverlap = round(Lsegm * overlap_perc/100)

        
        win=sig.windows.tukey(Lsegm,param)

                
       
        print('Lsegm',Lsegm)
        #print('length',len(Lsegm))
        #print('win',win)
        #print('len win',len(win))

        logging.info('after win')

        f,Pxx=sig.welch(s,Fs,win,Lsegm,noverlap=noverlap,detrend=False, return_onesided= not(d_s))
        #f,Pxx=sig.welch(np.real(s),Fs,noverlap=noverlap,detrend=False, return_onesided=True)

        logging.info('After wlsch func')

        
        y=10*np.log10(Pxx/max(Pxx))
        
        L = len(f)
        n1 = round((L-1)/Fs*fmin + (L+1)/2)
        n2 = round((L-1)/Fs * fmax + (L+1)/2)
        f_plot =f # f[n1-1:n2]
        y_plot =y # y[n1-1:n2]

        o=f_plot.argsort()
        f_plot = f_plot[o]
        y_plot = y_plot[o]

            


        
        if zoom_plt == 1:
            f_ctr_diff = np.absolute(f_plot -  f_centre)
            f_ctr_idx  = f_ctr_diff.argmin()

            f_ctr_max_diff = np.absolute(f_plot -  f_centre_max)
            f_ctr_max_idx  = f_ctr_max_diff.argmin()

            f_ctr_min_diff = np.absolute(f_plot -  f_centre_min)
            f_ctr_min_idx  = f_ctr_min_diff.argmin()

        if no_plot==0:
            fig,ax=plt.subplots()
            plt.plot(f_plot/(freq_scale),y_plot, label = str(legend))
            plt.grid(visible=True)
            plt.legend()
            plt.title("Signal spectrum")


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
                        #ax.bar((self.Fc+i*BW)/(freq_scale),10*np.log10(ac),BW/(freq_scale))
                        if no_plot==0:
                            if i==0:
                                pass
                                #ax.hlines(10*np.log10(ac_plot),(Fc+offset+(i*BWi-0.5*BW_conf[BW_idx]))/(freq_scale),(Fc+offset+(i*BWi+0.5*BW_conf[BW_idx]))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                #ax.text(x=(Fc+offset+(i*BWi-0.5*BW_conf[BW_idx]))/(freq_scale),y=10*np.log10(ac)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')
                            else:
                                ax.hlines(10*np.log10(ac_plot),(Fc+offset+(i*BWi-0.5*ACLR_BW[BW_idx]))/(freq_scale),(Fc+offset+(i*BWi+0.5*ACLR_BW[BW_idx]))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                ax.text(x=(Fc+offset+(i-0.5)*BWi)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')

                                # Does this get back those ACLR side band "lines"? with proper scaling -> Not this part
                                #ax.hlines(10*np.log10(ac_plot),(Fc+offset+(i*BWi-0.5*ACLR_BW[BW_idx]))/(freq_scale),(Fc+offset+(i*BWi+0.5*ACLR_BW[BW_idx]))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                #ax.text(x=(Fc+offset+(i-0.5)*BWi)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')

                    BW_idx+=1

   

            a=5
            

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
                    #ax.bar((self.Fc+i*BW)/(freq_scale),10*np.log10(ac),BW/(freq_scale))
                    if no_plot==0:
                        if i==0:
                            pass
                            #ax.hlines(10*np.log10(ac),(Fc+(i*BW-0.5*BW_conf))/(freq_scale),(Fc+(i*BW+0.5*BW_conf))/(freq_scale),label=str(ac),colors='r',zorder=10)
                            #ax.text(x=(Fc+(i*BW-0.5*BW_conf))/(freq_scale),y=10*np.log10(ac)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')
                        else:
                            # Does this then get back those red lines
                            if ACLR_offset==0:
                                ax.hlines(10*np.log10(ac_plot),(Fc+(i*BW-0.5*ACLR_BW))/(freq_scale),(Fc+(i*BW+0.5*ACLR_BW))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                ax.text(x=(Fc+(i-0.5)*BW)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')
                            else:
                                ax.hlines(10*np.log10(ac_plot),(Fc+(i*ACLR_offset-0.5*ACLR_BW))/(freq_scale),(Fc+(i*ACLR_offset+0.5*ACLR_BW))/(freq_scale),label=str(ac),colors='r',zorder=10)
                                ax.text(x=(Fc+(i-0.5)*ACLR_offset)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')

                            #ax.hlines(10*np.log10(ac_plot),(Fc+(i*BW-0.5*ACLR_BW))/(freq_scale),(Fc+(i*BW+0.5*ACLR_BW))/(freq_scale),label=str(ac),colors='r',zorder=10)
                            #ax.text(x=(Fc+(i-0.5)*BW)/(freq_scale),y=10*np.log10(ac_plot)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')


        if no_plot==0:
            if zoom_plt == 1:
                ax.set_xlim(f_plot[f_ctr_min_idx]/(freq_scale),f_plot[f_ctr_max_idx]/(freq_scale))
            if not isinstance(PSD_min,str):
                plt.ylim(PSD_min,10)
            plt.xlabel("Frequency ["+unit_txt+"]")
            plt.ylabel("PSD [dB]")
            plt.show(block=False)
        if no_plot==0:
            if hasattr(BW,"__len__") and len(BW)>1:
                return fig, ACLR

            if BW!=0:
                return fig, ACLR 
            return fig
        else:
            if hasattr(BW,"__len__") and len(BW)>1:
                return f_plot[o]/(freq_scale),y_plot[o], ACLR
            if BW!=0:
                return f_plot[o]/(freq_scale),y_plot[o], ACLR
            else:
                return f_plot[o]/(freq_scale),y_plot[o]


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





