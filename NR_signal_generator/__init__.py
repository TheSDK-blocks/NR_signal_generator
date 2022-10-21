#-------------------------------------------------------------------------------
#--Copyright (c) 2017  Aalto University
#--
#--Permission is hereby granted, free of charge, to any person obtaining a copy
#--of this software and associated documentation files (the "Software"), to deal
#--in the Software without restriction, including without limitation the rights
#--to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#--copies of the Software, and to permit persons to whom the Software is
#--furnished to do so, subject to the following conditions:
#--
#--The above copyright notice and this permission notice shall be included in all
#--copies or substantial portions of the Software.
#--
#--THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#--IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#--FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#--AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#--LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#--OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#--SOFTWARE.
#-------------------------------------------------------------------------------

# Structure, functions and almost all comments are copied from Matlab code done by Enrico Roverato
import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *
#from rtl import *
#from rtl.testbench import *
#from rtl.testbench import testbench as vtb
#from eldo import *
#from eldo.testbench import *
#from eldo.testbench import testbench as etb 
import pdb
import numpy as np
import scipy.signal as sig
from scipy import interpolate as inter
import matplotlib.pyplot as plt
class NR_signal_generator(thesdk): #rtl,eldo,thesdk
    @property
    def _classfile(self):
        return os.path.dirname(os.path.realpath(__file__)) + "/"+__name__
 
    def __init__(self,*arg): # ,BW,osr,Nsymb,qam_type,bits

            self.print_log(type='I', msg='Inititalizing %s' %(__name__)) 
            self.proplist = [ 'BW','BWP','osr','QAM','in_bits','include_time_vector','Fc_gen' ];    # Properties that can be propagated from parent

            self.IOS=Bundle()
            #self.IOS.Members['A']=IO() # Pointer for input data
            #self.IOS.Members['B']=IO() # Pointer for input data
            #self.IOS.Members['C']=IO() # Pointer for input data
            #self.IOS.Members['D']=IO() # Pointer for input data
            #self.IOS.Members['E']=IO() # Pointer for input data
            self.IOS.Members['in_dem']=IO() # Pointer for input data

            self.IOS.Members['out']= IO()# Pointer for output data
            #self.IOS.Members['Y']= IO()# Pointer for output data
            #self.IOS.Members['X']= IO()# Pointer for output data          
            #self.IOS.Members['W']= IO()# Pointer for output data
            #self.IOS.Members['V']= IO()# Pointer for output datai
            #self.IOS.Members['U']= IO()# Pointer for output data

            self.BWP = np.array([[[4,7,0,1]]])
            self.QAM = "64QAM"
            self.osr = 1
            self.BW = np.array([200e6])
            #self.in_bits = np.array([[np.random.randint(2,size=4752*6)]])
            self.in_bits = np.array([["max"]])

            self.seed=0
            self.include_time_vector=0
            self.Fc_gen=0

            self.BW_conf=[]
            self.ACLR_BW=[]
            self.model='py';             # Can be set externally, but is not propagated
            self.par= False              # By default, no parallel processingi
            self.queue= []               # By default, no parallel processing
            self.IOS.Members['control_write']= IO() 
            self.fil="on"
            self.equalizer="on"
            self.fil_len=100 
            self.norm="max"             # max = normalize I & Q separately, amp = normalize amplitude to one
                                        


            if len(arg)>=1:
                parent=arg[0]
                self.copy_propval(parent,self.proplist)
                self.parent =parent;








    def main_gen(self):
        #self.BW=self.IOS.Members['A'].Data # Bandwidth as int 5-400 MHz. Negative BW leaves gap with selected width
        #self.osr=self.IOS.Members['B'].Data # Oversampling ratio (integer)
        #self.BWP=self.IOS.Members['C'].Data # List of bandwidth parts per carrier
        #self.qam_type=self.IOS.Members['D'].Data # Modulation type (BPSK (Not in LTE standard), 4QAM, 16QAM, 64QAM or 256QAM)
        #self.bits=self.IOS.Members['E'].Data # Inputdata as binary list
        if not hasattr(self.BW,"__len__") :
            self.BW=np.array([self.BW])
        if not hasattr(self.BW,"size") :
            self.BW=np.array([self.BW])

        self.cnstl, self.gen_bits=self.genMultiQAM()
        if self.Fc_gen != 0:
            self.osr_based_on_Fc()
        self.s_struct, self.f_off=self.genMultiNRdownlink()
        if self.include_time_vector==1:
            self.IOS.Members['out'].Data=self.s_struct["s"] # output signal, matrix with  columns time, I signal, Q signal
        else:
            self.IOS.Members['out'].Data=np.transpose(np.vstack((self.s_struct["s"][:,1],self.s_struct["s"][:,2])))
        if len(self.BW_conf)==1:
            self.BW_conf=self.BW_conf[0]
        
        if len(self.ACLR_BW)==1:
            self.ACLR_BW=self.ACLR_BW[0]
        elif len(self.ACLR_BW)==0:
            self.ACLR_BW=0
        #self.IOS.Members['Y'].Data=self.s_struct["cnstl"] # generated resource grid with PSS and CRS 
        #self.IOS.Members['X'].Data=self.cnstl # generated constellation points 
   

        


    def run_gen(self,*arg):
        if len(arg)<0:
            self.rap=True
            self.queue=arg[0]
        
        if self.model=='py':
            self.main_gen()

        #del self.iofile_bundle


    def main_dem(self):
        self.rec_sig=self.IOS.Members['in_dem'].Data # Input signal as matrix descibed in main_gen()      
        if not hasattr(self.BW,"__len__") :
            self.BW=[self.BW]
        self.dem=self.demMultiNRdownlink()
        self.dem_bits, self.dem_cnstl_vec=self.MultiQAMtoBit()
        #self.IOS.Members['W'].Data=self.dem_bits # Outputdata as binary list 
        #self.IOS.Members['V'].Data=self.dem # Demodulated resource grid with PSS and CRS   
        #self.IOS.Members['U'].Data=self.dem_cnstl_vec # Demodulated contellation points 
   
    def run_EVM(self):
        if self.model=='py':
            self.main_EVM()

    def main_EVM(self):
        self.EVM, self.rxDataSymbols=self.measMultiEVMdownlink()

    def run_dem(self,*arg):
        if len(arg)<0:
            self.rap=TRue
            self.queue=arg[0]
        
        if self.model=='py':
            self.main_dem()

        #del self.iofile_bundle




    def osr_based_on_Fc(self):
        #pdb.set_trace()
        BW_vect_abs=np.abs(self.BW)
        BW_tot=np.sum(BW_vect_abs) # get total bandwidth (in Hz)
        Fs_estimate=np.ceil(BW_tot/2)
        req_fs=self.Fc_gen*2+Fs_estimate
        osr=np.ceil(req_fs/Fs_estimate)
        if osr==0:
            pass
        else:
            self.osr=osr




    def MultiQAMtoBit(self):
        """ Method for calculate binary data based on recieved constellation points for multiple carriers.


            Example
            -------
            self.MultiQAMtoBit()

        """

        qam_type=self.QAM
        dem=self.dem
        BW=self.BW
        BWP=self.BWP
        bits=[]
        dem_cnstl_vec=[]
        for i in range(0,len(dem)):
            temp1,temp2=self.QAMtoBit(cnstl=dem[i],BW=BW[i],BWP=BWP[i])
            bits.append(temp1)
            dem_cnstl_vec.append(temp2)

        return bits, dem_cnstl_vec

    def measMultiEVMdownlink(self):
        """ Method for calculating EVM based on generated constellation points and recieved constellation points for multiple carriers.



            Example
            -------
            self.measMultiEVMdownlink()

        """

        BW=self.BW
        BWP=self.BWP
        dem=self.dem
        cnstl=self.s_struct['cnstl']

        N_BW=len(cnstl)
        EVM=[]
        #EVM=np.zeros(N_BW)
        rxDataSymbols=[]
        # measure EVM separately for each constellation
        for i in range(0,N_BW):
            N_BWP=len(cnstl[i])
            EVM_BWP=np.zeros(N_BWP)
            rxDataSymbols_BWP=[]
            for j in range(0,N_BWP):
                if cnstl[i][j]!=[]:
                    EVM1,rxDataSymbols1=self.measEVMdownlink(BW=BW[i],BWP=BWP[i][j],cnstl=cnstl[i][j],dem=dem[i][j])
                    EVM_BWP[j]=EVM1
                    rxDataSymbols_BWP.append(rxDataSymbols1)

                else:
                    rxDataSymbols_BWP.append([])
            rxDataSymbols.append(rxDataSymbols_BWP)
            EVM.append(EVM_BWP)
        return EVM, rxDataSymbols





    def demMultiNRdownlink(self):
        """ Method for demodulate constellation points from recieved signal for multiple carriers.



            Example
            -------
            self.demMultiNRdownlink()

        """

        sign=self.rec_sig
        BW=self.BW
        BWP=self.BWP
        Fs=self.s_struct["Fs"]
        f_off=self.f_off
        FFRwpos=0
        N_BW=BW.size
        cnstl=[]
        #t=np.arange(0,sign.size)/Fs # initialize time vector (for mixing)
        #pdb.set_trace()
        if sign.shape[1]==2:
            s=sign[:,0]+1j*sign[:,1]
            t=t=np.arange(0,len(s))/self.s_struct["Fs"]
        else:
            t=sign[:,0]
            s=sign[:,1]+1j*sign[:,2]
        # demodulate carriers
        for i in range(0,N_BW):
            BWi=BW[i]
            if BWi>0:
                # mix current carrier so that it is centered at 0 Hz
                v_mixed=s*np.exp(-1j*2*np.pi*(self.Fc_gen+f_off[i])*t)
                osr=[]
                # calculate OSR of current carrier
                for j in range(0,len(BWP[i])):
                    dl_osrl=self.NRparameters(mu=BWP[i][j][0],BW=BW[i])
                    osr.append(dl_osrl["Fs"])
                osr=np.around(Fs/np.array(osr))

                v_filt=v_mixed
                if self.fil=='on': 
                    v_filt=self.NRfilter(Fs=Fs,raw_vector=v_mixed,BW=BWi,osr=max(osr))

                a=self.demNRdownlink(s=v_filt,BW=BWi,BWP=BWP[i],osr=osr)
                cnstl.append(a)
            else:
                cnstl.append([])
        return cnstl



    


    def genMultiNRdownlink(self):
        """ Method for generating signal from constellation points for multiple carriers.


            Example
            -------
            self.genMultiNRdownlink()

        """

        #NOFDMsym=cnstl[0][0].size
        BW=self.BW
        BWP=self.BWP
        #pdb.set_trace()
        tot_osr=self.osr
        cnstl=self.cnstl
        N_BW=BW.size
        NFFT=[]
        Fss=[]
        LCM=int(1)

        # get integers proportional to base sampling rates
        # get also their LCM (least common multiple)
        BW_vect_abs=np.abs(BW)
        maxsf=0
        slength=[]
        for i in range(0,N_BW):
            sub_NFFT=[]
            slen=[]
            sub_fs=[]
            if BW[i]!=0:
                    
                if hasattr(BWP[i],"__len__") :
                    for j in range(0,len(BWP[i])):
                        up=self.NRparameters(mu=BWP[i][j][0],BW=BW_vect_abs[i])
                        start=up["RB"]*BWP[i][j][2]
                        stop=up["RB"]*BWP[i][j][3]
                        if stop<start:
                            raise Exception("Wrong BW of BWP")
                            return None, None

                        sub_fs.append(up["Fs"])
                        LCM=np.lcm(LCM,int(up["Fs"]))
                        if BW[i]>0:
                            sub_NFFT.append(up["NFFT"])
                            N_ofdm1=np.ceil(BWP[i][j][1]/(7*2**BWP[i][j][0]))
                            slen.append(N_ofdm1*up["Nofdm1"]+up["Nofdm2"]*(BWP[i][j][1]-N_ofdm1))
                        if up["Fs"]>maxsf:
                            maxsf=up["Fs"]

                else:

                    BW_of_BWP=BW[i]
                    up2=self.NRparameters(mu=mu[i],BW=BW_of_BWP)
                    NFFT.append(up2["NFFT"])
                    LCM=np.lcm(LCM,int(up2["NFFT"]))
                    if up2["Fs"]>maxsf:
                        maxsf=up2["Fs"]
            else:
                sub_fs.append(0)
            NFFT.append(sub_NFFT)
            Fss.append(sub_fs)
            slength.append(slen)
            #NFFT=np.zeros(BW.size)

 
        # get integer proportional to overall sampling rate
        self.NFFT_debug=max(NFFT)
        NFFT_tot_min=0
        
        a=np.array_split(Fss,len(Fss))
        a=np.concatenate(a,axis=0)
        for i in range(0,len(Fss)):
            NFFT_tot_min+=max(a[i])
        
        NFFT_tot=tot_osr*LCM*np.ceil(NFFT_tot_min/LCM)
        # calculate individual oversampling ratios

        Fss=np.concatenate(a,axis=0)
        Fss=Fss[Fss!=0]
        osr=NFFT_tot*np.ones_like(Fss)/Fss
        max_len=[]
        
        ind=0
        for i in range(0,N_BW):
            if BW[i]>0:
                sub_len=[]
                for j in range(0,len(BWP[i])):
                    sub_len.append(osr[ind]*slength[i][j])
                    ind+=1
                max_len.append(sum(sub_len))
            else:
                ind+=1
           
        max_len=max(max_len)
        slength=max_len
        Fs=maxsf*min(osr)
        smatrix=np.zeros((int(slength+self.fil_len*max(osr)),int(N_BW)),complex)
        cnstlmatrix=[]
        # generate carriers
        osr_ind=0
        for i in range(0,N_BW):
            BWi=BW[i]
            if BWi>0:
                sub_cnstlmatrix=[]
                sub_smatrix=np.zeros((int(slength),1),complex)
                out=self.genNRdownlink(BW=BWi,BWP=BWP[i],osr=osr[osr_ind:int(osr_ind+len(BWP[i]))],cnstl=cnstl[i])
                cnstlmatrix.append(out["cnstl"])

                s=np.concatenate(out["s"])
                self.testvar=s
                if self.fil=='on': 
                    s=np.pad(s,(0,(int(slength)-len(s))),constant_values=0) 
                    s=self.NRfilter(Fs=Fs,raw_vector=s,BW=BWi,osr=max(osr[osr_ind:int(osr_ind+len(BWP[i]))]))
                smatrix[0:len(s),i]=self.normalize(x=s,opt="max",k=1)
                #smatrix[0:len(s),i]=s
                osr_ind=int(osr_ind+len(BWP[i]))
            else:
                cnstlmatrix.append([])
                osr_ind=int(osr_ind+len(BWP[i]))
 
        s_raw=np.zeros(int(slength+self.fil_len*max(osr)),complex)
        BWtot=np.sum(BW_vect_abs) # get total bandwidth (in Hz)

        # mix carriers to proper frequency offset
        f_off=np.zeros(N_BW)
        t_vect=np.arange(0,slength+self.fil_len*max(osr))/Fs
        #plt.figure()
        for i in range(0,N_BW):
            BWi=BW[i]
            
            if BWi>0:
                f_off[i]=np.sum(BW_vect_abs[0:i])+BWi/2-BWtot/2
                #f_off[i]=
                test=smatrix[:,i]*np.exp(1j*2*np.pi*f_off[i]*t_vect)
                s_raw=s_raw+smatrix[:,i]*np.exp(1j*2*np.pi*(self.Fc_gen+f_off[i])*t_vect)

                #fig=plot_PSD2(test,100,Fs)
        #plt.grid()
        #plt.title("Carriers separately")
        #plt.show()
        #plt.figure()
        #fig=plot_PSD2(s_raw[:int(len(s_raw)/2)],100,Fs)
        #plt.grid()
        #plt.title("First frame")
        #plt.show()
        #plt.figure()
        #fig=plot_PSD2(s_raw[int(len(s_raw)/2):],100,Fs)
        #plt.grid()
        #plt.title("Second frame")
        #plt.show()
        if (self.norm == "max"):
            s=self.normalize(x=s_raw,opt="max",k=1) # normalize final signal to 1
        elif (self.norm == "amp"):
            s=self.normalize(x=s_raw,opt="amp",k=1) # normalize final signal to 1

        #s=s_raw
        output_format=np.transpose(np.vstack((t_vect, np.real(s),np.imag(s))))
        out={
        "s":output_format,
        "cnstl":cnstlmatrix,
        "Fs":Fs,
        
        }

        return out,  f_off



    def genMultiQAM(self):
        """ Method for generating QAM constellation points based on input bits for multiple carriers.


            Example
            -------
            self.genMultiQAM()

        """
        #pdb.set_trace()
        BW=self.BW
        BWP=self.BWP
        qam_type=self.QAM
        bits=self.in_bits
        
        N_BW=BW.size
        cnstl=[]
        gen_bits=[]
        for i in range(0,N_BW):
            BWi=BW[i]
            if BWi>0:
                sub_cnstl=[]
                sub_bits=[]
                b=np.size(bits[i],0)
                for j in range(0,b):
                    a,bit=self.genQAM(bits=bits[i][j],BW=BW[i], BWP=BWP[i][j])
                    sub_cnstl.append(a)
                    sub_bits.append(bit)
                cnstl.append(sub_cnstl)
                gen_bits.append(sub_bits)
            else:
                cnstl.append([])
                gen_bits.append([])
        return cnstl , gen_bits


    def spline_inter(self,**kwargs):
        """ Method for calculating spline interpolation for resource block sizeing.

            Parameters
            ----------
            mu : integer (0,1,2,3,4) 
               5G NR numerology
            BW : integer
                Bandwidth of carrier.

            Example
            -------
            self.spline_inter(mu=1,BW=10e6)

        """

        mu=kwargs.get('mu')
        BW=kwargs.get('BW')


        if mu==0:   # Numbers of RB from standard 38.101
            x=[0,5e6,10e6,15e6,20e6,25e6,40e6,50e6]
            y=[0,25,52,79,106,133,216,270]
        elif mu==1:
            x=[0,5e6,10e6,15e6,20e6,25e6,40e6,50e6,60e6,80e6,100e6]
            y=[0,11,24,38,51,65,106,133,162,217,273]
        elif mu==2:
            x=[0,10e6,15e6,20e6,25e6,40e6,50e6,60e6,80e6,100e6,200e6]
            y=[0,11,18,24,31,51,65,79,107,135,264]
        elif mu==3:
            x=[0,50e6,100e6,200e6,400e6]
            y=[0,32,66,132,264]
        elif mu==4:
            x=[0,100e6,200e6,400e6]
            y=[0,32,64,128]
        
        if mu==4:
            p=inter.interp1d(x,y)
        else:
            p=inter.interp1d(x,y,kind='cubic')   
        
        try:
            RB=int(np.floor(p(BW)))
        except:
            raise Exception("Width of selected BWP is too big for selected mu")
        

        
        return RB
 

    def NRparameters(self,**kwargs):
        """ Method for calculating necessary parameters

            Parameters
            ----------
            mu : integer (0,1,2,3,4) 
               5G NR numerology
            BW : integer
                Bandwidth of carrier
            osr : integer
                Oversampling factor

            Example
            -------
            self.NRparameters(mu=1, BW=10e6, osr= 1)

        """

        mu=kwargs.get('mu')
        BW=kwargs.get('BW')       
        gen=kwargs.get('gen',0)       
        N_slot_in_subframe=2**mu
        SCS=2**mu*15e3
        
        min_BW=20*SCS*12
         

        RB=self.spline_inter(mu=mu,BW=BW)
        if gen==1:
            self.BW_conf.append(RB*12*SCS)
        temp1=[0,5e6,10e6,15e6,20e6,25e6,40e6,50e6,60e6,80e6,100e6,200e6,400e6]
        temp2=[0, 25*12*15e3, 52*12*15e3, 79*12*15e3, 106*12*15e3, 133*12*15e3, 216*12*15e3, 270*12*15e3,  162*12*30e3,  217*12*30e3,  273*12*30e3,   264*12*60e3,  264*12*120e3, ]
        if BW in temp1 and gen==1:
            self.ACLR_BW.append(temp2[np.where(temp1==BW)[0][0]])

        if BW<min_BW:
            raise Exception("Width of selected BWP is too small for selected mu")
            return None
        NFFT = 2**np.ceil(np.log2(RB*12/0.85))    # FFT size
        NFFT=max(128,NFFT)
        Tc=1/(15e3*2**mu*NFFT)
        Ts=1/(15e3*2048)
        k=Ts/Tc
        Fs = NFFT*SCS  # sampling frequency
        Ncp1 = 144*k*(1/(2**mu))+16*k      # length of cyclic prefix 0 and 7*2**mu
        Ncp2 = 144*k*(1/(2**mu))    # length of cyclic prefixes else
        Nofdm1 = NFFT+Ncp1  # length of OFDM symbol 0
        Nofdm2 = NFFT+Ncp2  # length of OFDM symbols 1-6
     
        if 'osr' in kwargs:
            osr=kwargs.get('osr')
        else:
            osr=1

        # set parameters that are not proportional to BW:
        # - W     = EVM window length
        # - Lroll = optimum symbol rolloff length to keep EVM < 1%
        # - RB    = number of Resource Blocks
        Lroll=0
        if NFFT==128:
            Lroll=4
        elif NFFT==256:
            Lroll=6
        elif NFFT==512:
            Lroll=4
        elif NFFT==1024:
            Lroll=6
        elif NFFT==2048:
            Lroll=8
        if Lroll==0:
            Lroll=max(0,8-2*(11-(np.log2(NFFT))))
        up={
        "Fs":Fs*osr,
        "NFFT":NFFT*osr,
        "Ncp1":Ncp1*osr,
        "Ncp2":Ncp2*osr,
        "Nofdm1":Nofdm1*osr,
        "Nofdm2":Nofdm2*osr,
        #"Nslot":Nslot*osr,
        #"W":W*osr,
        "RB":RB,
        "Nsc":RB*12, # number of occupied subcarriers
        "Lroll":Lroll
        }
        
        return up
        
    def genPSS(self,**kwargs):
        """ Method for generation Primary Synchronization  Signal.

            Parameters
            ----------
            id2 : integer
                Second cell ID defined for 5G NR

            Example
            -------
            self.genPSS(id2=0)

        """

        N_ID_2=kwargs.get('id2')

        # generate m-sequence

        n=np.arange(0,127)
        m=np.mod(n+43*N_ID_2,127)

        #arr=[]
        #for i in range(0,len(m)):
        #    arr.append(self.x(i))
        #d_PSS=1-2*np.array(arr)
    
        d_PSS=np.array( [ 1, -1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1,  1,  1, -1, \
            1, -1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1,  1 ,-1, -1,  1,\
            -1, -1 , 1, -1,  1, -1, -1, -1,  1, -1,  1,  1 , 1, -1, -1,\
            1,  1, -1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, \
            1, -1, -1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, \
            -1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, 1, -1,  1,  1,  1,  1, -1])
        d=d_PSS[m]
        return d



    #def x(self,m):
    #    if (m == 6) or (m == 5) or (m == 4) or (m == 2) or (m == 1):
    #        return 1
    #    elif (m == 0) or (m == 3):
    #        return 0
    #    else:
    #        i=m-7
    #        return (self.x(i + 4) + self.x(i)) % 2  

    
    def genDMRS(self,**kwargs):
        """ Method for generation Demodulation Reference Signal

            Parameters
            ----------
            N_ID_cell : integer 
               Cell ID
            Nsymb : integer
                Number of OFDM symbols
            RB : integer
                Number of used Resource Blocks

            Example
            -------
            self.genDMRS(N_ID_cell=0, Nsymb=14, RB=5)

        """

        N_ID_cell=kwargs.get('N_ID_cell')
        Nsymb=kwargs.get('Nsymb')
        RB=kwargs.get('RB')


        #N_RB_maxDL = 110
        N_RB=int(RB*12/2)
        r = np.zeros((N_RB, int(Nsymb)),complex);
        for i in range(0,int(Nsymb)):
            # get slot number & symbol number within the slot
            ns=np.floor(i/14)
            l=i%14

            c_init=np.mod((2**17*(14*ns+l+1)*(2*N_ID_cell+1)+2*N_ID_cell),2**31)
            c=self.rnd3GPPGenerator(c_init=c_init, Mpn=2*N_RB)
            # create reference-signal sequence
            ones=np.ones(int(len(c)/2))
            a=np.reshape(np.array(c),(-1,2))
            b=a[0,:]
            r[:,i]=np.add((1/(np.sqrt(2)))*np.subtract(ones,2*a[:,0]),1j*(1/(np.sqrt(2)))*np.subtract(ones,2*a[:,1]))
            
        return r
        
    def rnd3GPPGenerator(self,**kwargs):
        """ Method for calculation of pseude-random sequence for 5G NR

            Parameters
            ----------
            c_init : integer 
                Initial value used for sequence generation
            Mpn : integer
                Second initial value that sets length of the sequence

            Example
            -------
            self.calc_EVM(c_init=5,Mpn=30)

        """

        c_init=kwargs.get('c_init')
        Mpn=kwargs.get('Mpn')


        Nc=1600
        x1=np.zeros(Mpn+Nc)
        x2=np.zeros(Mpn+Nc)
        x1[0]=1
        c_init_bit=np.binary_repr(int(c_init),31)
        for i in range(0,31):
            x2[i]=float(c_init_bit[30-i])
        # calculate m-sequences x1 and x2
        for i in range(0,Mpn+Nc-31):
            x1[i+31]=(x1[i+3]+x1[i])%2
            x2[i+31]=(x2[i+3]+x2[i+2]+x2[i+1]+x2[i])%2
        c=np.zeros(Mpn)
        # calculate pseudo-random sequence
        for n in range(0,Mpn):
            c[n]=(x1[n+Nc]+x2[n+Nc])%2
        return c


    def genQAM(self,**kwargs):
        """ Method for generating constellation point based on input bits.

            Parameters
            ----------
            bits : array of binary values 
               Binary input data
            BW : integer
                Bandwidth of carrier
            BWP : array of certain structure
                Bandwidth part of corresponding carrier. [a,b,c,d]
                where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
                d=highest used frequency of bandwidth in %, 0=<c<d=<1.    
            Example
            -------
            self.genQAM(bits=[1,0...1,2],BW=10e6, BWP=[0,14,0,1])

        """

        bits=kwargs.get('bits')
        BW=kwargs.get('BW')
        BWP=kwargs.get('BWP')


        qam_type=self.QAM
        qam=np.array([],complex)
        dl=self.NRparameters(mu=BWP[0],BW=BW)
        Nsymb=int(BWP[1])
        RE=np.full((dl["Nsc"],Nsymb),None)
        start=int(np.floor(dl["RB"]*BWP[2])*12)
        stop=int(np.ceil(dl["RB"]*BWP[3])*12)
        
        RE[:start,:]=0
        RE[stop:,:]=0
            
        # set DMRS symbols to zero
        k=np.arange(start,stop,2)
            
        for l in range(0,int(Nsymb)):
            if l%14==2 :
                RE[k,l]=0

        # set PSS symbols to zero
        #if n==0:
        l=min([Nsymb,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
        k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
        first=int(k[0])
        last=int(k[-1])
        RE[first:last+1,l]=0
        

        unusedOFDM=np.transpose(np.where(RE==None))
        unusedOFDM=unusedOFDM[np.lexsort((unusedOFDM[:,0], unusedOFDM[:,1]))]
        Ncnstl=len(unusedOFDM)
        #pdb.set_trace()
        if bits=="max":
            np.random.seed(self.seed)
        if qam_type=="16QAM":
            M=16
            if bits=="max":
                bits=np.random.randint(2,size=int(np.log2(M)*Ncnstl))
                
            if  len(bits)>np.log2(M)*Ncnstl:
                 raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0) 
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(10)*((1-2*bits[4*i])*(2-(1-2*bits[4*i+2]))+1j*(1-2*bits[4*i+1])*(2-(1-2*bits[4*i+3])))
                qam=np.append(qam,d)
    

        elif qam_type=="4QAM":
            M=4
            if bits=="max":
                bits=np.random.randint(2,size=int(np.log2(M)*Ncnstl))
            if  len(bits)>np.log2(M)*Ncnstl:
                raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")
            else:
                if  len(bits)!=np.log2(M)*Ncnstl:
                    bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0)
                for i in range(0,int(len(bits)/np.log2(M))):
                    d=1/np.sqrt(2)*((1-2*bits[2*i])+1j*(1-2*bits[2*i+1]))
                    qam=np.append(qam,d)

        elif qam_type=="BPSK":
            M=2
            if bits=="max":
                bits=np.random.randint(2,size=int(np.log2(M)*Ncnstl))

            if  len(bits)>np.log2(M)*Ncnstl:
                raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0) 
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(2)*((1-2*bits[i])+1j*(1-2*bits[i]))
                qam=np.append(qam,d)

        elif qam_type=="256QAM":
            M=256
            if bits=="max":
                bbits=np.random.randint(2,size=int(np.log2(M)*Ncnstl))

            if  len(bits)>np.log2(M)*Ncnstl:
                 raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0)
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(170)*((1-2*bits[8*i])*(8-(1-2*bits[8*i+2])*(4-(1-2*bits[8*i+4])*(2-(1-2*bits[8*i+6]))))+1j*(1-2*bits[8*i+1])*(8-(1-2*bits[8*i+3])*(4-(1-2*bits[8*i+5])*(2-(1-2*bits[8*i+7])))))
                qam=np.append(qam,d)
            
        else : # 64QAM
            M=64
            if bits=="max":
                bits=np.random.randint(2,size=int(np.log2(M)*Ncnstl))

            if  len(bits)>np.log2(M)*Ncnstl:
                 raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0)
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(42)*((1-2*bits[6*i])*(4-(1-2*bits[6*i+2])*(2-(1-2*bits[6*i+4])))+1j*(1-2*bits[6*i+1])*(4-(1-2*bits[6*i+3])*(2-(1-2*bits[6*i+5]))))  
                qam=np.append(qam,d)
        
        
        return qam,bits


    def QAMtoBit(self,**kwargs):
        """ Method for generating binary array based on constellation points.

            Parameters
            ----------
            cnstl : array  
               Array of constellation points
            BW : integer
                Bandwidth of carrier
            BWP : array of certain structure
                Bandwidth part of corresponding carrier. [a,b,c,d]
                where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
                d=highest used frequency of bandwidth in %, 0=<c<d=<1.    
            Example
            -------
            self.QAMtoBit(cnstl=[0.2+i*0.5,...,1-i*0,7], BW=10e6, BWP=[1,7,0,1])

        """

        cnstl=kwargs.get('cnstl')
        BW=kwargs.get('BW')
        BWP=kwargs.get('BWP')

        qam_type=self.QAM
        cnstl_of_carrier=[]
        vector=[]
        
        for n in range(0,len(cnstl)):
            if cnstl ==[]:
                cnstl_of_carrier.append([])
                vector.append([])
            
            
            mu=BWP[n][0]
            dl=self.NRparameters(mu=mu,BW=BW) # get NR parameters
            N_ID_1 = 0 # physical-layer cell-identity group
            N_ID_2 = 0 # physical-layer identity within the group
            N_ID_cell = 3*N_ID_1 + N_ID_2 # cell identity
            # initialize data constellations (non-normalized)
            DataSymbols_nnorm = np.copy(cnstl[n])
            start=int(np.floor(dl["RB"]*BWP[n][2])*12)
            stop=int(np.ceil(dl["RB"]*BWP[n][3])*12)
            Nsymb=int(BWP[n][1])
            
            DataSymbols_nnorm[:start,:]=0
            DataSymbols_nnorm[stop:,:]=0

            # set DMRS symbols to zero
            k=np.arange(start,stop,2)
            
            for l in range(0,int(Nsymb)):
                if l%14==2 :
                    DataSymbols_nnorm[k,l]=0

                    
            # set PSS symbols to zero
            #if n==0:
            l=min([Nsymb,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
            k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
            first=int(k[0])
            last=int(k[-1])
            DataSymbols_nnorm[first:last+1,l]=0
            


            # vectorize constellations
            DataSymbols_vect = np.reshape(DataSymbols_nnorm,(DataSymbols_nnorm.size),order="F" );

        
            # keep only non-zero symbols

            DataSymbols_nzero = DataSymbols_vect[np.argwhere(DataSymbols_vect)]
            bit_vect=[]
            if qam_type=="16QAM":
                M=16
                m=np.log2(M)
            elif qam_type=="4QAM":
                M=4
                m=np.log2(M)
            elif qam_type=="BPSK":
                M=2
                m=np.log2(M)
            elif qam_type=="256QAM":
                M=256
                m=np.log2(M)
            else : # 64QAM
                M=64
                m=np.log2(M)
            points=[]
            points_as_bits=[]
            for i in range(0,M):
                bits=np.array(list(np.binary_repr(i,int(m))),dtype=int)
                if qam_type=="16QAM":
                    d=1/np.sqrt(10)*((1-2*bits[0])*(2-(1-2*bits[2]))+1j*(1-2*bits[1])*(2-(1-2*bits[3])))
                elif qam_type=="4QAM":
                    d=1/np.sqrt(2)*((1-2*bits[0])+1j*(1-2*bits[1]))
    

                elif qam_type=="BPSK":
                    d=1/np.sqrt(2)*((1-2*bits[0])+1j*(1-2*bits[0]))

                elif qam_type=="256QAM":
                    d=1/np.sqrt(170)*((1-2*bits[0])*(8-(1-2*bits[2])*(4-(1-2*bits[4])*(2-(1-2*bits[6]))))+1j*(1-2*bits[1])*(8-(1-2*bits[3])*(4-(1-2*bits[5])*(2-(1-2*bits[7])))))
            
                else : # 64QAM
                    d=1/np.sqrt(42)*((1-2*bits[0])*(4-(1-2*bits[2])*(2-(1-2*bits[4])))+1j*(1-2*bits[1])*(4-(1-2*bits[3])*(2-(1-2*bits[5]))))
                points.append(d)
                points_as_bits.append(bits)
            for j in range(0,len(DataSymbols_nzero)):
                test=np.array(points)-DataSymbols_nzero[j]
                index = np.argmin(np.abs(np.array(points)-DataSymbols_nzero[j]))
                bit_vect.extend(points_as_bits[index])
                

            cnstl_of_carrier.append(DataSymbols_nzero)
            vector.append(np.array(bit_vect))
        return vector, cnstl_of_carrier
        
    def genNRdownlink(self,**kwargs):
        """ Method for generating 5G NR signal based on constellation points.

            Parameters
            ----------
            BW : integer
                Bandwidth of carrier
            BWP : array of certain structure
                Bandwidth part of corresponding carrier. [a,b,c,d]
                where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
                d=highest used frequency of bandwidth in %, 0=<c<d=<1.
            osr : integer
                Oversampling factor
            cnstl : array
                Array of constellation points
            Example
            -------
            self.genNRdownlink(BW=10e6, BWP=[2,7,0,1],osr=1, cnstl=[1-i*0.6,...,-0,6+i*0.3])

        """

        BW=kwargs.get('BW')
        BWP=kwargs.get('BWP')
        osr=kwargs.get('osr')
        cnstl=kwargs.get('cnstl')

        signal=[]
        REs=[]
        for n in range(0,len(BWP)):
            mu=BWP[n][0]
            dl=self.NRparameters(mu=mu,BW=BW,osr=osr[n],gen=1)
            #N_RB_maxDL = 110  # largest downlink bandwidth (in Resource Blocks)
            N_ID_1 = 0  # physical-layer cell-identity group
            N_ID_2 = 0  # physical-layer identity within the group
            N_ID_cell = 3*N_ID_1 + N_ID_2 # cell identity
            Nsymb=int(BWP[n][1])
            N_slots=np.ceil(Nsymb/(7*2**mu)) # get total number of slots to be generated (round up)

            Nsym_cp2 = Nsymb - N_slots
            Nsamples = dl["Nofdm1"]*N_slots + dl["Nofdm2"]*Nsym_cp2 # get total number of samples

            #RE=np.copy(cnstl)
            RE=np.full((dl["Nsc"],Nsymb),None)
            start=int(np.floor(dl["RB"]*BWP[n][2])*12)
            stop=int(np.ceil(dl["RB"]*BWP[n][3])*12)
            BW_of_BWP=(stop-start)*(15e3*2**mu)
            dl2=self.NRparameters(mu=mu,BW=BW_of_BWP,osr=osr[n])  # To check if BWP is more than 20 RB
            RE[:start,:]=0
            RE[stop:,:]=0
            #RE=np.zeros((dl["Nsc"],Nsymb),complex)
            symb_in_frame=int(2**mu*10*14)
            for i in range(0,int(np.ceil(Nsymb/symb_in_frame))):
                if (Nsymb-i*symb_in_frame)>symb_in_frame:
                    NsymbInGrid=symb_in_frame
                else:
                    NsymbInGrid=(Nsymb-i*symb_in_frame)
                
                r_DMRS=self.genDMRS(N_ID_cell=N_ID_cell,Nsymb=NsymbInGrid,RB=(stop-start)/12) # DeModulation Reference Signals
                
                # map DMRS

                k=np.arange(start,stop,2)
            
                for l in range(0,int(NsymbInGrid)):
                    if l%14==2 :

                        RE[k,i*symb_in_frame+l]=r_DMRS[:,l]
               
            # map PSS
            d_PSS=self.genPSS(id2=N_ID_2) # PRIMARY SYNCHRONIZATION SIGNAL
            #if n==0:
            l=min([NsymbInGrid,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
            k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
            first=int(k[0])
            last=int(k[-1])
            RE[first:last+1,i*symb_in_frame+l]=np.concatenate((np.zeros(56),d_PSS,np.zeros(57))) # set  edge subcarriers to 0
        

            unusedOFDM=np.transpose(np.where(RE==None))
            unusedOFDM=unusedOFDM[np.lexsort((unusedOFDM[:,0], unusedOFDM[:,1]))]

            if len(cnstl[n])>len(unusedOFDM):
                raise Exception("Not enoigh OFDM symbols")
                #print("Not enough OFDM symbols")
                return None

            for i in range(0,len(cnstl[n])):
                l=unusedOFDM[i][1]
                k=unusedOFDM[i][0]
                RE[k,l]=cnstl[n][i]
        
            RE[RE==None]=0+0*1j
            RE=RE.astype(complex)
  
        
            # map resource elements to modulated subcarriers
            subcarriers=np.zeros((int(dl["NFFT"]),Nsymb),complex)
            subcarriers[int(dl["NFFT"]-dl["Nsc"]/2):int(dl["NFFT"])]=RE[0:int(dl["Nsc"]/2)]
            subcarriers[0:int(dl["Nsc"]/2)]=RE[int(dl["Nsc"]/2):]
        

            OFDMsymbols=np.transpose(np.fft.ifft(np.transpose(subcarriers))) # convert the subcarrier spectrum to time domain
            ext_symbols=np.concatenate((OFDMsymbols,OFDMsymbols,OFDMsymbols)) # extend symbols prior to windowing
            # set up Tukey window
            Lr=dl["Lroll"]*osr[n] # symbol rolloff length accounting for osr
            w_rise=0.5*(1+np.cos(np.pi*np.arange(Lr+1,2*Lr+1)/Lr)) # window start
            w_fade=0.5*(1+np.cos(np.pi*np.arange(0,Lr)/Lr)) # window end
            # create Tukey window for the first OFDM symbol of each slot
            w_ofdm1=np.zeros(int(3*dl["NFFT"]),complex)
            w_ofdm1[int(dl["NFFT"]-dl["Ncp1"]-Lr):int(dl["NFFT"]-dl["Ncp1"])]=w_rise # window start
            w_ofdm1[int(2*dl["NFFT"]-1):int(2*dl["NFFT"]+Lr-1)]=w_fade # window end
            w_ofdm1[int(dl["NFFT"]-dl["Ncp1"]):int(2*dl["NFFT"]-1)] = 1 # window center
            # create Tukey window for OFDM symbols 2 to 7 of each slot
            w_ofdm2to7=np.zeros(int(3*dl["NFFT"]),complex)
            w_ofdm2to7[int(dl["NFFT"]-dl["Ncp2"]-Lr):int(dl["NFFT"]-dl["Ncp2"])] = w_rise;  
            w_ofdm2to7[int(2*dl["NFFT"]-1):int(2*dl["NFFT"]+Lr-1)] = w_fade  
            w_ofdm2to7[int(dl["NFFT"]-dl["Ncp2"]):int(2*dl["NFFT"]-1)] = 1 

            win_symbols=np.zeros((int(3*dl["NFFT"]),Nsymb),complex)
            # apply windowing to OFDM symbols
            for i in range(1,Nsymb+1):
                    if (i==1 or (i%(7*2**mu))==1): # use first Tukey window
                        win_symbols[:,i-1]=ext_symbols[:,i-1]*w_ofdm1
                    else: # use second Tukey window
                        win_symbols[:,i-1]=ext_symbols[:,i-1]*w_ofdm2to7
            # align OFDM symbols on the rows of a matrix (REALLY cryptic code!)
            xdim=Nsymb+2
            ydim=dl["Nofdm1"]+Nsamples+3*dl["NFFT"]
            sum_matrix=np.zeros((int(xdim),int(ydim)),complex)
            sum_matrix[0,0:int(3*dl["NFFT"])]=win_symbols[:,-1] # place front-end OFDM symbol

            for i in range(0,Nsymb):
                long_cp=np.floor(i/(7*2**mu)) # current slot (in samples)
                short_cp=i-long_cp # current OFDM symbol in the slot
                i1=int(dl["Nofdm1"]+long_cp*dl["Nofdm1"]+short_cp*dl["Nofdm2"]+1) # start index
                i2=int(dl["Nofdm1"]+long_cp*dl["Nofdm1"]+short_cp*dl["Nofdm2"]+3*dl["NFFT"]) # end index
                sum_matrix[i+1,i1-1:i2]=win_symbols[:,i] # place symbol

            sum_matrix[Nsymb+1,int(-3*dl["NFFT"]):] = win_symbols[:,0] # place back-end OFDM symbol

            sum_vector=sum_matrix.sum(axis=0) # sum each column of the matrix
            raw_vector=sum_vector[int(2*dl["NFFT"]):int(2*dl["NFFT"]+Nsamples)] # discard tails of SC-FDMA vector
            
            final_vector=raw_vector
           
            #s=self.normalize(final_vector,"max",1) # normalize output signal to 1
            s=final_vector

            signal.append(s)
            REs.append(RE)
        out={
        "s":signal,
        "cnstl":REs,
        "Fs":dl["Fs"]
        }

        return out




        
    def normalize(self,**kwargs):
        """ Method for normalizing signal.

            Parameters
            ----------
            x : array 
               Signal
            opt : sting
                Normalization option (max, pow)
            k : float
                Input is normalized to this value
                - if option = "max", then max(|Re(yi)|, |Im(yi)|) = k
                - if option = "pow" or "totpow", then the statistical power of y  is k^2

            Example
            -------
            self.normalize(x=[0.3,...,0.7], opt="max",k=1)

        """

        x=kwargs.get('x')
        opt=kwargs.get('opt')
        k=kwargs.get('k')

        # INPUTS:
        # x = input vector or matrix
        # opt = can be either "max", "pow", or "totpow"
        # k = input is normalized to this factor
        #     - if option = "max", then max(|Re(yi)|, |Im(yi)|) = k
        #     - if option = "pow" or "totpow", then the statistical power of y (or
        #       each of its columns) is k^2
        if opt=="max":
            test_matrix=np.concatenate((np.absolute(np.real(x)),np.absolute(np.imag(x))))
            max_value=test_matrix.max()
            y=k*x/max_value

        elif opt == "amp":
            amp = np.sqrt((np.real(x)**2) + (np.imag(x)**2))
            max_amp = max(amp)
            y = k*x/max_amp

        elif opt=="pow":
            pow=(np.absolute(x)**2).sum()/len(x)
            y=k*x/np.sqrt(pow)

        elif opt=="totpow":
            """TBD if needed"""
        return y

    def NRfilter(self,**kwargs):
        """ Method for filtering signal.

            Parameters
            ----------
            Fs : integer 
               Sampling frequency
            raw_vector : array
                Signal to be filtered
            BW : integer
                Bandwidth
            osr : integer
                Oversampling factor
            Example
            -------
            self.NRfilter(Fs=5e9,raw_vector=[0.3,...,0.7], opt="max",k=1)

        """

        Fs=kwargs.get('Fs')
        raw_vector=kwargs.get('raw_vector')
        BW=kwargs.get('BW')

        # get sampling frequency

        Fs=int(Fs)
        if 'osr' in kwargs:
            osr=kwargs.get('osr')
        else:
            osr=1
        # select filter parameters
        fac=1
        #fac=BW/150e6
        order=self.fil_len*np.ceil(osr*fac) # filter order

        dF=0
        for mu in range(0,5):
            try:
                up=self.NRparameters(mu=(4-mu),BW=BW)
                if ((up["RB"])*12*15e3*2**(4-mu))>dF:
                    dF=(up["RB"])*12*15e3*2**(4-mu)
            except:
                continue
        dF=(BW-dF)*0.5
        
        Fc=BW/2 # center of transition bandwidth
        t_vect=np.arange(0,len(raw_vector))/Fs
        # filter NR signal with circular convolution
        #pdb.set_trace()
        b=sig.remez(int(order+1),[0,Fc-dF,Fc+dF,0.5*Fs],[1,0],Hz=int(Fs)) 
        #plt.figure()
        #plt.plot(b)
        #plt.show()
        #w, h = sig.freqz(b, [1], worN=2000)
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(0.5*Fs*w/np.pi, 20*np.log10(np.abs(h)))
        #ax.set_xlim(0, 0.5*Fs)
        #plt.show()
        bb=b
        #bb=np.pad(b,(0,len(raw_vector)-len(b)),constant_values=0)
        #s_fil=np.fft.ifft(np.fft.fft(raw_vector,len(raw_vector))*np.fft.fft(bb,len(raw_vector)))
        s_fil=np.convolve(raw_vector.reshape((-1,1))[:,0],bb,mode='full').reshape((-1,1))

        # compensate group delay (order/2)
        s_out=np.concatenate((s_fil[int(order/2):],s_fil[0:int(order/2)]))
        #s_out=s_fil
        return s_out[:,0]
        


    
    def demNRdownlink(self,**kwargs):
        """ Method for demodulation signal.

            Parameters
            ----------
            s : array 
               Signal
            BW : integer
                Bandwidth
            BWP : array of certain structure
                Bandwidth part of corresponding carrier. [a,b,c,d]
                where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
                d=highest used frequency of bandwidth in %, 0=<c<d=<1.
            osr : integer
                Oversampling factor

            Example
            -------
            self.demNRdownlink(s=[0.3,...,0.7], BW=10e6,BWP=[1,14,0,1], osr=1)

        """

        s=kwargs.get('s')
        BW=kwargs.get('BW')
        BWP=kwargs.get('BWP')
        osr=kwargs.get('osr')

        FFTwpos=0
        end_of_prev_sig=0
        equalize=self.equalizer
        cnstl_of_carrier=[]
        #if self.fil=="on":
 
        #    #sign=self.NRfilter(sign,BW,osr)
        for n in range(0,len(BWP)):
            mu=BWP[n][0]
            N_symb_TOT=int(BWP[n][1])
            Nsym_cp1=np.ceil(N_symb_TOT/(7*2**mu)) 

            
            # get various parameters related to input signal
            dl=self.NRparameters(mu=mu,BW=BW,osr=osr[n]) # general NR parameters
            Nsym_cp2 = N_symb_TOT - Nsym_cp1
            N_sampl = int(dl["Nofdm1"]*Nsym_cp1 + dl["Nofdm2"]*Nsym_cp2) # get total number of samples
            if len(BWP)>1:
                sign=s[end_of_prev_sig:end_of_prev_sig+N_sampl]
                end_of_prev_sig=end_of_prev_sig+N_sampl
            else:
                sign=s
                N_sampl=len(s)
            #if self.fil=="on":
            #    sign=self.NRfilter(mu,sign,BW,osr[n])
            N_slots=np.floor(N_symb_TOT/14) # number of slots (integer)
            N_ID_1=0    # physical-layer cell-identity group
            N_ID_2=0    # physical-layer identity within the group
            N_ID_cell=3*N_ID_1+N_ID_2 # cell identity
            # generate PSS in the time-domain
            d_PSS=self.genPSS(id2=N_ID_2) # generate PSS sequence
            # map PSS sequence to subcarriers
            PSSf=np.zeros((int(dl["NFFT"])),complex)
            PSSf[-64:]=d_PSS[0:64]
            PSSf[1:64]=d_PSS[64:127]
            PSSt=np.fft.ifft(PSSf) # convert to time-domain
            # find match of PSS within the input vector, by computing cross-correlation
            # and looking for its maximum
            pad=np.pad(PSSt, (0,len(sign)-len(PSSt)), constant_values=0)
            xcmax=np.argmax(np.absolute(np.correlate(sign,pad,"full"))) 
            #pdb.set_trace()
            # calculate ideal result of the cross-correlation maximum (see above)
            if N_symb_TOT==0:
                print("ERROR")
                return 0
            elif N_symb_TOT==1:
                xcmax_id = N_sampl + dl["Ncp1"]
            elif N_symb_TOT < 4:
                xcmax_id = N_sampl + dl["Nofdm1"] + (N_symb_TOT-2)*dl["Nofdm2"] + dl["Ncp2"]
            else:
                xcmax_id = N_sampl + dl["Nofdm1"] + 2*dl["Nofdm2"] + dl["Ncp2"]

            # circularly shift input vector so that PSS location becomes "as expected"
            invect_aligned = np.roll(sign,int(xcmax_id-xcmax)-1)

            # calculate shift of FFT window from "nominal" position (i.e. when CP is
            # completely discarded)
            #invect_aligned[:]=1
            FFTw_shift_1 = -dl["Ncp1"]/2 + FFTwpos
            FFTw_shift_2 = -1*np.ceil(dl["Ncp2"]/2) + FFTwpos
            OFDMmatrix = np.zeros((int(dl["NFFT"]), int(N_symb_TOT)),complex)
            # iterate through the OFDM symbols
            for index in range(0,int(N_symb_TOT)):

                cp1=np.ceil(index/(7*2**mu)) 

                cp2 = index -cp1
                if index%(7*2**mu)==0:
                    i1 = dl["Nofdm1"]*cp1 + dl["Nofdm2"]*cp2 + dl["Ncp1"] + FFTw_shift_1 + 1
                    i2 = dl["Nofdm1"]*cp1 + dl["Nofdm2"]*cp2 + dl["Ncp1"] + FFTw_shift_1 + dl["NFFT"]
                else:
                    i1 = dl["Nofdm1"]*cp1 + dl["Nofdm2"]*cp2 + dl["Ncp2"] + FFTw_shift_2 + 1
                    i2 = dl["Nofdm1"]*cp1 + dl["Nofdm2"]*cp2 + dl["Ncp2"] + FFTw_shift_2 + dl["NFFT"]
                OFDMmatrix[:, index] = invect_aligned[int(i1-1):int(i2)]

            # calculate compensation factor, due to shift of FFT window from "nominal"
            # position (i.e. when CP is completely discarded)
            k=np.arange(0,dl["NFFT"])
            e = np.outer(np.exp(-1j*2*np.pi*k*FFTw_shift_2/dl["NFFT"]) ,np.ones((int(N_symb_TOT))))
            for index in np.arange(0,int(N_symb_TOT),(7*2**mu)):
                e[:, int(index)] = np.exp(-1j*2*np.pi*k*FFTw_shift_1/dl["NFFT"])
        
            # calculate FFT and apply compensation factor
            subcarriers = np.multiply(np.transpose(np.fft.fft(np.transpose(OFDMmatrix))) , e)
            # map subcarriers to resource elements
            RE = np.zeros((int(dl["Nsc"]), int(N_symb_TOT)),complex)
            RE[0:int(dl["Nsc"]/2),:] = subcarriers[int(-dl["Nsc"]/2):,:]
            RE[int(dl["Nsc"]/2):,:] = subcarriers[0:int(dl["Nsc"]/2),:]
            # if equalization is deactivated, return at this point already
            if equalize=='off':
                cnstl = RE;
                cnstl_of_carrier.append(cnstl)
            # re-create DMRS/PSS grid (i.e. post-FFT ideal reference signal)
            RE_id = np.ones((RE.shape),complex)
            start=int(np.floor(dl["RB"]*BWP[n][2])*12)
            stop=int(np.ceil(dl["RB"]*BWP[n][3])*12)
            symb_in_frame=int(2**mu*10*14)
            for i in range(0,int(np.ceil(N_symb_TOT/symb_in_frame))):
                if (N_symb_TOT-i*symb_in_frame)>symb_in_frame:
                    NsymbInGrid=symb_in_frame
                else:
                    NsymbInGrid=(N_symb_TOT-i*symb_in_frame)
                
                r_DMRS=self.genDMRS(N_ID_cell=N_ID_cell,Nsymb=NsymbInGrid,RB=(stop-start)/12) # DeModulation Reference Signals
                
                # map DMRS
                #v_shift=N_ID_cell%6
           
                k=np.arange(start,stop,2)
            
                for l in range(0,int(NsymbInGrid)):
                    if l%14==2 :

                        RE_id[k,i*symb_in_frame+l]=r_DMRS[:,l]
               
            d_PSS=self.genPSS(id2=N_ID_2) # PRIMARY SYNCHRONIZATION SIGNAL
            #if n==0:
            l=min([NsymbInGrid,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
            k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
            first=int(k[0])
            last=int(k[-1])
            RE_id[first:last+1,i*symb_in_frame+l]=np.concatenate((np.zeros(56),d_PSS,np.zeros(57))) # set  edge subcarriers to 0

            # calculate the complex ratios of the post-FFT acquired signal "RE" and the
            # post-FFT ideal signal "RE_id", for each reference symbol
            complex_ratios = np.divide(RE, RE_id)
            a = np.absolute(complex_ratios)
            phi = np.angle(complex_ratios)
            #  unwrap phase of complex ratios at symbol #0 of each slot
            l=np.arange(0,N_symb_TOT,dtype=int)
            l=l[l%14==2]
            k1=np.arange(start,stop,2)
            for k in k1:
                for i in  np.arange(1,l.size):
                    delta_phi = phi[k,int(l[i])] - phi[k,int(l[i-1])]
                    if np.absolute(delta_phi) >= np.pi:
                        phi[k,l[i:]] = phi[k,l[i:]] - 2*np.pi*np.sign(delta_phi)


            # perform time averaging at each reference signal subcarrier of the complex
            # ratios (in TS 36.104 the time-averaging length is 10 subframes, here for
            # simplicity the time-averaging length is that of the signal)
            a_avg = np.zeros((int(dl["Nsc"])))
            phi_avg = np.zeros((int(dl["Nsc"])))

            l=np.arange(0,N_symb_TOT,dtype=int)
            l=l[l%14==2]
            k1=np.arange(start,stop,2)

            for k in k1:
                test=a[int(k),l]
                a_avg[int(k)] = np.mean(a[int(k),l])
                phi_avg[int(k)] = np.mean(phi[int(k),l])


            # the equalizer coefficients for amplitude and phase "a_coeff" and
            # "phi_coeff" at the reference signal subcarriers are obtained by computing
            # the moving average in the frequency domain of the time-averaged reference
            # signal subcarriers, i.e. every third subcarrier (or sixth, if less than 5
            # OFDM symbols)
            a_coeff =np.zeros((int(dl["Nsc"])))
            phi_coeff = np.zeros((int(dl["Nsc"])))
            k_PSS=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
            k=np.arange(start,stop,2)
        
            if N_symb_TOT == 3: # exclude 5+5 null reference subcarriers around the PSS
                for i in np.concatenate((range(0,56),range(182,239))):
                    if np.any(k+1 == k_PSS[i]):
                        ind_k_to_remove = np.argwhere(k+1 == k_PSS[i])
                        k = k[np.concatenate((np.arange(0,ind_k_to_remove-1),np.arange(ind_k_to_remove,-1)))]

            for i in np.arange(1,k.size+1):
                m_avg_w_length = min(2*i-1, 2*(k.size-i)+1, 19)
                m_avg_imp_resp = (np.ones((1, m_avg_w_length))) / m_avg_w_length
                test=np.dot(m_avg_imp_resp , a_avg[k[int(i-np.floor(m_avg_w_length/2)-1):int(i+np.floor(m_avg_w_length/2))]])
                a_coeff[k[int(i-1)]] = np.dot(m_avg_imp_resp , a_avg[k[int(i-np.floor(m_avg_w_length/2)-1):int(i+np.floor(m_avg_w_length/2))]])
                phi_coeff[k[int(i-1)]] = np.dot(m_avg_imp_resp , phi_avg[k[int(i-np.floor(m_avg_w_length/2)-1):int(i+np.floor(m_avg_w_length/2))]])
        
             # perform linear interpolation to compute coefficients for each subcarrier
            a_coeff = np.interp(np.arange(1,dl["Nsc"]+1), k+1, a_coeff[k] )
            phi_coeff = np.interp(np.arange(1,dl["Nsc"]+1), k+1, phi_coeff[k] )

            cnstl = np.zeros((RE.shape),complex)
            # equalize resource elements and return
            for i in np.arange(0,dl["Nsc"]):
                cnstl[i,:]=RE[i,:]/(a_coeff[i]*np.exp(1j*phi_coeff[i]))

            cnstl_of_carrier.append(cnstl)


        return cnstl_of_carrier




    def measEVMdownlink(self,**kwargs):
        """ Method for calculation EVM.

            Parameters
            ----------
            BW : integer
                Bandwidth
            BWP : array of certain structure
                Bandwidth part of corresponding carrier. [a,b,c,d]
                where a=numerology,b=number of OFDM symbols, c=lowest used frequency of bandwidth in %
                d=highest used frequency of bandwidth in %, 0=<c<d=<1.
            cnstl : array
                Generated constellation points used as reference
            dem : array
                Recieved constellation points
            Example
            -------
            self.measEVMdownlink(BW=10e6,BWP=[1,14,0,1],cnstl=[1-i*0.6,...,-0,6+i*0.3],dem=[1-i*0.6,...,-0,6+i*0.3])

        """
        BW=kwargs.get('BW')
        BWP=kwargs.get('BWP')
        cnstl=kwargs.get('cnstl')
        dem=kwargs.get('dem')

        

        Nsc,N_symb=cnstl.shape # get some info from constellation matrix dimensions
        mu=BWP[0]
        if dem==[]:
            return 0,0
        Nsc_rx, N_symb_rx = dem.shape
        if Nsc!=Nsc_rx or   N_symb!=N_symb_rx:
            return 0,0
        dl=self.NRparameters(mu=mu,BW=BW) # get NR parameters
        N_ID_1 = 0 # physical-layer cell-identity group
        N_ID_2 = 0 # physical-layer identity within the group
        N_ID_cell = 3*N_ID_1 + N_ID_2 # cell identity
        # initialize data constellations (non-normalized)
        rxDataSymbols_nnorm = np.copy(dem)
        refDataSymbols_nnorm = np.copy(cnstl)
        # set DMRS symbols to zero
        start=int(np.floor(dl["RB"]*BWP[2])*12)
        stop=int(np.ceil(dl["RB"]*BWP[3])*12)
        rxDataSymbols_nnorm[:start,:]=0
        rxDataSymbols_nnorm[stop:,:]=0
        refDataSymbols_nnorm[:start,:]=0
        refDataSymbols_nnorm[stop:,:]=0
        k=np.arange(start,stop,2)
            
        for l in range(0,int(N_symb)):
            if l%14==2 :
                rxDataSymbols_nnorm[k,l]=0
                refDataSymbols_nnorm[k,l]=0
        
        # set PSS symbols to zero
        l=min([N_symb,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
        k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
        first=int(k[0])
        last=int(k[-1])
        rxDataSymbols_nnorm[first:last+1,l]=0
        refDataSymbols_nnorm[first:last+1,l]=0

        # vectorize constellations
        rxDataSymbols_vect = np.reshape(rxDataSymbols_nnorm,(rxDataSymbols_nnorm.size),order="F" );
        refDataSymbols_vect = np.reshape(refDataSymbols_nnorm,(refDataSymbols_nnorm.size),order="F" );

        # keep only non-zero symbols
        rxDataSymbols_nzero = rxDataSymbols_vect[np.argwhere(rxDataSymbols_vect)]
        refDataSymbols_nzero = refDataSymbols_vect[np.argwhere(refDataSymbols_vect)]
        # normalize constellation powers
        #rxDataSymbols = self.normalize(rxDataSymbols_nzero, 'pow', 1);
        #refDataSymbols = self.normalize(refDataSymbols_nzero, 'pow', 1);
        rxDataSymbols = rxDataSymbols_nzero
        refDataSymbols = refDataSymbols_nzero

        # calculate EVM using Matlab's built-in functions
        #Copied from https://github.com/TheSDK-blocks/f2_testbench/blob/master/f2_testbench/analyzers_mixin.py
        
        reference=refDataSymbols
        received=rxDataSymbols
        
        # Takes zeros into account
        pad=np.zeros(np.shape(received),complex)
        pad[:np.shape(reference)[0],:np.shape(reference)[1]]=reference
        reference=pad
        
        # Do not take zeros into account
        #received=np.delete(received,range(len(reference),len(received)))


        #Shape the vectors: time is row observation is column
        reference.shape=(-1,1)
        received.shape=(-1,1)

        #RMS for Scaling
        rmsref=np.std(reference)
        rmsreceived=np.std(received)
        EVM=(np.mean(np.mean(np.abs(received-reference)**2,axis=0)/np.mean(np.abs(reference)**2,axis=0)))**(1/2)

        
        return EVM, rxDataSymbols
        


        





    def define_io_conditions(self):
        
        # Input A is read to verilog simulation after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['A']._io_condition='initdone'
        # Output is read to verilog simulation when all of the outputs are valid, 
        # and after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['out'].verilog_io_condition_append(cond='&& initdone')

def plot_PSD(x,a,*arg):
    if len(arg)>=1:
        Fs=int(arg[0])
        s=x.s_struct["s"]
        s=s[:,1]+1j*s[:,2]
        BW=x.BW[0]
    else:
        Fs = int(x.s_struct["Fs"])
        s=x.s_struct["s"]
        s=s[:,1]+1j*s[:,2]
        BW=x.BW
    


    #s=x.NRfilter(Fs,s,BW,x.osr)
    Lsegm_perc = 10
    Fs_given = 0
    plot_color = 'k'
    win_type = 'tukey'
    param = 0.1
    overlap_perc = 50

    Fs_given = 1
    
    fmin = -Fs/2
    fmax = Fs/2

    Lsegm_perc = a
    a=len(s)
    Lsegm = round(len(s)*Lsegm_perc/100)
    noverlap = round(Lsegm * overlap_perc/100)
    win=sig.tukey(Lsegm,param)
    f,Pxx=sig.welch(s,Fs,win,Lsegm,noverlap=noverlap,detrend=False)
    
    y=10*np.log10(Pxx/max(Pxx))


    
    L = len(f)
    n1 = round((L-1)/Fs*fmin + (L+1)/2)
    n2 = round((L-1)/Fs * fmax + (L+1)/2)
    f_plot = f[n1-1:n2]
    y_plot = y[n1-1:n2]
    fig=plt.figure()
    plt.plot(f_plot/(10**6),y_plot)
    plt.grid()
    plt.title("Signal spectrum")
    plt.show(block=False)
    return fig


def plot_PSD2(x,a,*arg):
    if len(arg)>=1:
        Fs=int(arg[0])
        s=x
        
    else:
        Fs = int(x.s_struct["Fs"])
        s=x
        
    

    #s=x.NRfilter(s,BW,x.osr)
    Lsegm_perc = 10
    Fs_given = 0
    plot_color = 'k'
    win_type = 'tukey'
    param = 0.1
    overlap_perc = 50

    Fs_given = 1
    
    fmin = -Fs/2
    fmax = Fs/2

    Lsegm_perc = a
    a=len(s)
    Lsegm = round(len(s)*Lsegm_perc/100)
    noverlap = round(Lsegm * overlap_perc/100)
    win=sig.tukey(Lsegm,param)
    f,Pxx=sig.welch(s,Fs,win,Lsegm,noverlap=noverlap,detrend=False)
    
    y=10*np.log10(Pxx/max(Pxx))


    
    L = len(f)
    n1 = round((L-1)/Fs*fmin + (L+1)/2)
    n2 = round((L-1)/Fs * fmax + (L+1)/2)
    f_plot = f[n1-1:n2]
    y_plot = y[n1-1:n2]
    #fig=plt.figure()
    plt.plot(f_plot/(10**6),y_plot)
    #plt.grid()
    #plt.show()
    #return fig



def meas(x,Fs):
    s=x.s_struct["s"]
    s=s[:,1]+1j*s[:,2]
    f,Pxx=sig.periodogram(s,Fs,return_onesided=False)

    y=Pxx
    fmin = -Fs/2
    fmax = Fs/2
    L = len(f)
    n1 = int(round((L-1)/Fs*fmin + (L+1)/2))
    n2 = int(round((L-1)/Fs * fmax + (L+1)/2))
    f_plot = f[n1-1:n2]
    y_plot = y[n1-1:n2]
    plt.figure()
    #plt.pcolormesh(t, f, Sxx)
    plt.plot(f_plot/(10**6),y_plot)
    plt.grid()
    plt.show()


if __name__=="__main__":
    #import matplotlib.pyplot as plt
    #from  NR_signal_studio import *
    #from  NR_signal_studio.controller import controller as NR_signal_studio_controller

    #length=1024
    #method="multi"
    #method="single"
    #BWP=np.array([[[0,14,0,1],[0,14,0,1]],[[0,14,0,1],[1,14,0,1],[2,14,0,1]]])  #[mu, symbols, BW_low(0...1),BW_high(0...1)>BW_low]
    #BWP=np.array([[[4,7,0,1]],[[4,7,0,1]]])
    BWP=np.array([[[4,7,0,1]]])
    #BW=np.array([400e6,400e6])
    BW=np.array([200e6])
    #mu=[0,0]
    #QAM="16QAM"
    QAM="64QAM"
    #QAM="256QAM"
    #QAM="4QAM"
    #QAM="BPSK" # Not for downlink
    #bits=np.array([0,0,0,1]*3816)

    #bits=np.array(list(bits)*(int(240/4))) # same as ones in matlab nsymb 2 16qam
    #bits=np.array(list(bits)*(int(6000)))
    #bits=np.random.randint(2,size=1028*2) 
    #bits=np.random.randint(2,size=1710*6)  
    #bits2=np.random.randint(2,size=4752*6)
    #bits3=np.random.randint(2,size=3738*6)
    #bits=np.random.randint(2,size=1710*4) 
    #in_bits=[bits]*len(BW)
    #in_bits =np.array([[bits2]])
    in_bits=np.array([["max"]])
    #in_bits=np.array([["max"],['max']])
    osr=1
    Fc=0#1e9
    if not hasattr(BW,"__len__") :
        a=5
    elif  hasattr(BW,"__len__") :
        test=NR_signal_generator()
        #test.IOS.Members['in_dem']=test.IOS.Members['out']
        test.BW=BW
        test.BWP=BWP
        test.QAM=QAM
        test.osr=osr
        test.Fc_gen=Fc
        #test.include_time_vector=1
        test.in_bits=in_bits
        test.run_gen()
        #pdb.set_trace()
        rand=np.random.rand(1000,2)/10000
         
        test.IOS.Members['in_dem'].Data=np.vstack([rand,test.IOS.Members['out'].Data])
        test.run_dem()
        test.run_EVM()
        a=np.transpose(np.vstack((test.s_struct["s"][:,0],test.s_struct["s"][:,1])))
        #np.savetxt('signal.csv',test.s_struct["s"],delimiter=',')

        plt.figure()
        plt.plot(test.s_struct["s"][:,0],test.s_struct["s"][:,1])
        plt.plot(test.s_struct["s"][:,0],test.s_struct["s"][:,2])
        plt.show(block=False)
        print(test.EVM)
        for i in range(0,BW.size):
            for j in range(0,len(BWP[i])):
                if  test.EVM[i].any():
                    if test.EVM[i][j]!=0:
                        plt.figure()
                        plt.plot(test.rxDataSymbols[i][j].real,test.rxDataSymbols[i][j].imag,'o')
                        plt.plot(test.cnstl[i][j].real,test.cnstl[i][j].imag,'o')
                        plt.title("Carrier "+str(i+1)+", frame "+str(j+1))
                        plt.show(block=False)
    
    #meas(test,test.s_struct["Fs"])
    a=plot_PSD(test,100,test.s_struct["Fs"])
    print("tst")
    input()
