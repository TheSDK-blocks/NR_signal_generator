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
import cmd
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

#from thesdk import *
#from rtl import *
#from rtl.testbench import *
#from rtl.testbench import testbench as vtb
#from eldo import *
#from eldo.testbench import *
#from eldo.testbench import testbench as etb 
#from module1 import spectrum_analyzer
import pdb
import numpy as np
import scipy.signal as sig
import scipy.interpolate as inter
import matplotlib.pyplot as plt
class NR(): #rtl,eldo,thesdk
    def __init__(self,BW,BWP,osr,qam_type,bits,*arg): 

            self.bits=bits
            if not hasattr(BW,"__len__") :
                self.BW=[BW]
            else:
                self.BW=BW
            self.osr=osr #integer oversampling ratio
            self.fil="on"
            self.BWP=BWP
            self.fil_len=100
            self.equalizer="on"
            self.qam_type=qam_type #constellation type ('16QAM' or '64QAM')
            self.cnstl, self.gen_bits=self.genMultiQAM(self.BW, self.BWP, self.qam_type,self.bits)
            self.s_struct, self.f_off=self.genMultiNRdownlink(self.BW,self.BWP,self.osr,self.cnstl)
            self.rec_s=None
            self.dem=self.demMultiNRdownlink(self.s_struct["s"],self.BW,self.BWP,self.s_struct["Fs"],self.f_off)
            self.dem_bits, self.dem_cnstl_vec=self.MultiQAMtoBit(self.qam_type,self.dem, self.BW,self.BWP)
            self.EVM,self.rxDataSymbols=self.measMultiEVMdownlink(self.BW, self.BWP, self.s_struct["cnstl"],self.dem) 

            a=6



    def MultiQAMtoBit(self,qam_type,dem,BW,BWP):
        bits=[]
        dem_cnstl_vec=[]
        for i in range(0,len(dem)):
            temp1,temp2=self.QAMtoBit(qam_type,dem[i],BW[i],BWP[i])
            bits.append(temp1)
            dem_cnstl_vec.append(temp2)

        return bits, dem_cnstl_vec

    def measMultiEVMdownlink(self,BW,BWP,cnstl,dem):

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
                    EVM1,rxDataSymbols1=self.measEVMdownlink(BW[i],BWP[i][j],cnstl[i][j],dem[i][j])
                    EVM_BWP[j]=EVM1
                    rxDataSymbols_BWP.append(rxDataSymbols1)

                else:
                    rxDataSymbols_BWP.append([])
            rxDataSymbols.append(rxDataSymbols_BWP)
            EVM.append(EVM_BWP)
        return EVM, rxDataSymbols




    def demMultiNRdownlink(self,sign,BW,BWP,Fs,f_off):
        FFRwpos=0
        N_BW=BW.size
        cnstl=[]
        #t=np.arange(0,sign.size)/Fs # initialize time vector (for mixing)
        t=sign[:,0]
        s=sign[:,1]+1j*sign[:,2]
        # demodulate carriers
        for i in range(0,N_BW):
            BWi=BW[i]
            if BWi>0:
                # mix current carrier so that it is centered at 0 Hz
                v_mixed=s*np.exp(-1j*2*np.pi*f_off[i]*t)
                osr=[]
                # calculate OSR of current carrier
                for j in range(0,len(BWP[i])):
                    dl_osrl=self.NRparameters(BWP[i][j][0],BW[i])
                    osr.append(dl_osrl["Fs"])
                osr=np.around(Fs/np.array(osr))

                v_filt=v_mixed
                if self.fil=='on': 
                    v_filt=self.NRfilter(Fs,v_mixed,BWi,max(osr))

                a=self.demNRdownlink(v_filt,BWi,BWP[i],osr)
                cnstl.append(a)
            else:
                cnstl.append([])
        return cnstl



    



    def genMultiNRdownlink(self,BW,BWP,tot_osr,cnstl):
        #NOFDMsym=cnstl[0][0].size
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
            if BW[i]>0:
                    
                if hasattr(BWP[i],"__len__") :
                    for j in range(0,len(BWP[i])):
                        up=self.NRparameters(BWP[i][j][0],BW_vect_abs[i])
                        start=up["RB"]*BWP[i][j][2]
                        stop=up["RB"]*BWP[i][j][3]
                        if stop<start:
                            raise Exception("Wrong BW of BWP")
                            return None, None

                        sub_NFFT.append(up["NFFT"])
                        sub_fs.append(up["Fs"])
                        LCM=np.lcm(LCM,int(up["Fs"]))
                        N_ofdm1=np.ceil(BWP[i][j][1]/(7*2**BWP[i][j][0]))
                        slen.append(N_ofdm1*up["Nofdm1"]+up["Nofdm2"]*(BWP[i][j][1]-N_ofdm1))

                        if up["Fs"]>maxsf:
                            maxsf=up["Fs"]

                else:

                    BW_of_BWP=BW[i]
                    up2=self.NRparameters(mu[i],BW_of_BWP)
                    NFFT.append(up2["NFFT"])
                    LCM=np.lcm(LCM,int(up2["NFFT"]))
                    if up2["Fs"]>macsf:
                        maxsf=up2["Fs"]
            else:
                sub_fs.append(0)
            NFFT.append(sub_NFFT)
            Fss.append(sub_fs)
            slength.append(slen)
            #NFFT=np.zeros(BW.size)

     
        #LCM=int(1)
        #for i in range(0,N_BW):
        #    up=self.NRparameters(0,BW_vect_abs[i])
        #    NFFT[i]=int(up["NFFT"])
        #    LCM=np.lcm(LCM,int(up["NFFT"]))
        # get integer proportional to overall sampling rate
        NFFT_tot_min=0
        #a=np.array_split(NFFT,len(NFFT))
        a=np.array_split(Fss,len(Fss))
        a=np.concatenate(a,axis=0)
        for i in range(0,len(Fss)):
            NFFT_tot_min+=max(a[i])
        #a=np.array_split(NFFT,len(NFFT))
        #a=np.concatenate(a,axis=0)
        #a=np.array(np.meshgrid([i for i in a]))#.T.reshape(-1,len(NFFT))
        #NFFT_tot_min=max(a)
        NFFT_tot=tot_osr*LCM*np.ceil(NFFT_tot_min/LCM)
        # calculate individual oversampling ratios

        Fss=np.concatenate(a,axis=0)
        Fss=Fss[Fss!=0]
        osr=NFFT_tot*np.ones_like(Fss)/Fss
        max_len=[]
        #neg=0
        ind=0
        for i in range(0,N_BW):
            if BW[i]>0:
                sub_len=[]
                for j in range(0,len(BWP[i])):
                    sub_len.append(osr[ind]*slength[i][j])
                    ind+=1
                max_len.append(sum(sub_len))
            #else:
            #    neg=neg+1
        max_len=max(max_len)
        #up=self.NRparameters(BWP[0][0][0],BW_vect_abs[0],osr[0][0])
        #N_ofdm1=np.ceil(NOFDMsym/7)
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
                out=self.genNRdownlink(BWi,BWP[i],osr[osr_ind:int(osr_ind+len(BWP[i]))],cnstl[i])
                cnstlmatrix.append(out["cnstl"])
                s=np.concatenate(out["s"])
                self.testvar=s
                if self.fil=='on': 
                    s=np.pad(s,(0,(int(slength)-len(s))),constant_values=0) 
                    s=self.NRfilter(Fs,s,BWi,max(osr[osr_ind:int(osr_ind+len(BWP[i]))]))
                smatrix[0:len(s),i]=self.normalize(s,"max",1)
                #smatrix[0:len(s),i]=s
                osr_ind=int(osr_ind+len(BWP[i]))
            else:
                cnstlmatrix.append([])
 
        s_raw=np.zeros(int(slength+self.fil_len*max(osr)),complex)
        BWtot=np.sum(BW_vect_abs) # get total bandwidth (in Hz)

        # mix carriers to proper frequency offset
        f_off=np.zeros(N_BW)
        t_vect=np.arange(0,slength+self.fil_len*max(osr))/Fs
        plt.figure()
        for i in range(0,N_BW):
            BWi=BW[i]
            
            if BWi>0:
                f_off[i]=np.sum(BW_vect_abs[0:i])+BWi/2-BWtot/2
                test=smatrix[:,i]*np.exp(1j*2*np.pi*f_off[i]*t_vect)
                s_raw=s_raw+smatrix[:,i]*np.exp(1j*2*np.pi*f_off[i]*t_vect)

                fig=plot_PSD2(test,100,Fs)
        plt.grid()
        plt.title("Carriers separately")
        plt.show(block=False)
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
        s=self.normalize(s_raw,"max",1) # normalize final signal to 1
        #s=s_raw
        output_format=np.transpose(np.vstack((t_vect, np.real(s),np.imag(s))))
        out={
        "s":output_format,
        "cnstl":cnstlmatrix,
        "Fs":Fs,
        
        }

        return out,  f_off




    def genMultiQAM(self,BW, BWP, qam_type,bits):

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
                    a,bit=self.genQAM(qam_type,bits[i][j],BW[i], BWP[i][j])
                    sub_cnstl.append(a)
                    sub_bits.append(bit)
                cnstl.append(sub_cnstl)
                gen_bits.append(sub_bits)
            else:
                cnstl.append([])
                gen_bits.append([])
        return cnstl , gen_bits


    def spline_inter(self,mu,BW):
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
        
        #xnew=np.linspace(min(x), max(x), num=100, endpoint=True)
        #plt.plot(x,y,'o', xnew, p(xnew))
        #plt.show()
        
        return RB
 

    def NRparameters(self,mu,BW,*arg):
       
        N_slot_in_subframe=2**mu
        SCS=2**mu*15e3
        
        min_BW=20*SCS*12
        

        RB=self.spline_inter(mu,BW)
        #RB2=int(np.floor(BW/(SCS*12)))
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
        #Nslot = 2*Nofdm1+12*Nofdm2  # slot length

        if len(arg)<1:
            osr=1
        else:
            osr=int(arg[0])

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
        # calculate extremities of EVM window
        #excl=up["Ncp2"]-up["W"] # number of CP samples not included in W
        #up["n_l"]=up["W"]+excl/2 # start for EVMl
        #up["n_h"]=excl/2 # start for EVMh
        #up["n_b"]=int(round(up["Ncp2"]/2)) # start for EVMb ("best EVM")
        return up
        
    def genPSS(self,N_ID_2):

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

    
    def genDMRS(self,N_ID_cell,Nsymb,RB):
        #N_RB_maxDL = 110
        N_RB=int(RB*12/2)
        r = np.zeros((N_RB, int(Nsymb)),complex);
        for i in range(0,int(Nsymb)):
            # get slot number & symbol number within the slot
            ns=np.floor(i/14)
            l=i%14

            c_init=np.mod((2**17*(14*ns+l+1)*(2*N_ID_cell+1)+2*N_ID_cell),2**31)
            c=self.rnd3GPPGenerator(c_init, 2*N_RB)
            # create reference-signal sequence
            ones=np.ones(int(len(c)/2))
            a=np.reshape(np.array(c),(-1,2))
            b=a[0,:]
            r[:,i]=np.add((1/(np.sqrt(2)))*np.subtract(ones,2*a[:,0]),1j*(1/(np.sqrt(2)))*np.subtract(ones,2*a[:,1]))
            
        return r
        
    def rnd3GPPGenerator(self,c_init, Mpn):
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


    def genQAM(self,qam_type,bits, BW, BWP):

        qam=np.array([],complex)
        dl=self.NRparameters(BWP[0],BW)
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
        if qam_type=="16QAM":
            M=16
            if  len(bits)>np.log2(M)*Ncnstl:
                 raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0) 
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(10)*((1-2*bits[4*i])*(2-(1-2*bits[4*i+2]))+1j*(1-2*bits[4*i+1])*(2-(1-2*bits[4*i+3])))
                qam=np.append(qam,d)
    

        elif qam_type=="4QAM":
            M=4
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
            if  len(bits)>np.log2(M)*Ncnstl:
                raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0) 
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(2)*((1-2*bits[i])+1j*(1-2*bits[i]))
                qam=np.append(qam,d)

        elif qam_type=="256QAM":
            M=256
            if  len(bits)>np.log2(M)*Ncnstl:
                 raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0)
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(170)*((1-2*bits[8*i])*(8-(1-2*bits[8*i+2])*(4-(1-2*bits[8*i+4])*(2-(1-2*bits[8*i+6]))))+1j*(1-2*bits[8*i+1])*(8-(1-2*bits[8*i+3])*(4-(1-2*bits[8*i+5])*(2-(1-2*bits[8*i+7])))))
                qam=np.append(qam,d)
            
        else : # 64QAM
            M=64
            if  len(bits)>np.log2(M)*Ncnstl:
                 raise Exception("Not enough OFDM symbols. Max "+str(int(np.log2(M)*Ncnstl))+" bits or "+str(Ncnstl)+ " constellation points")

            if  len(bits)!=np.log2(M)*Ncnstl:
                bits=np.pad(bits,(0,int(np.log2(M)*Ncnstl-len(bits))),constant_values=0)
            for i in range(0,int(len(bits)/np.log2(M))):
                d=1/np.sqrt(42)*((1-2*bits[6*i])*(4-(1-2*bits[6*i+2])*(2-(1-2*bits[6*i+4])))+1j*(1-2*bits[6*i+1])*(4-(1-2*bits[6*i+3])*(2-(1-2*bits[6*i+5]))))  
                qam=np.append(qam,d)
        
        
        return qam,bits


    def QAMtoBit(self,qam_type,cnstl,BW,BWP):
        cnstl_of_carrier=[]
        vector=[]
        
        for n in range(0,len(cnstl)):
            if cnstl ==[]:
                cnstl_of_carrier.append([])
                vector.append([])
            #Nsc,N_symb=cnstl.shape # get some info from constellation matrix dimensions
            ## determine system bandwidth from number of subcarriers
            #if Nsc==72:
            #     BW = 1.4e6;
            #elif  Nsc==180:
            #    BW = 3e6
            #elif  Nsc== 300:
            #    BW = 5e6
            #elif  Nsc== 600:
            #    BW = 10e6
            #elif  Nsc== 900:
            #    BW = 15e6
            #elif  Nsc== 1200:
            #    BW = 20e6
            #else:
            #    return [], []
            mu=BWP[n][0]
            dl=self.NRparameters(mu,BW) # get NR parameters
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

            #v_shift=N_ID_cell%6
            #m=np.arange(0,2*dl["RB"])
            #for l in np.arange(0,N_symb):
            #    if l%7==0 or l%7==4:
            #        v=3*((l%7)!=0)
            #        k=6*m+(v+v_shift)%6
            #        DataSymbols_nnorm[k,l] = 0
        
            # set PSS symbols to zero
            #if n==0:
            l=min([Nsymb,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
            k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
            first=int(k[0])
            last=int(k[-1])
            DataSymbols_nnorm[first:last+1,l]=0
            
            #l = min(N_symb, 7)  
            #k= np.arange(0,72) - 31 + int(dl["Nsc"]/2) - 5
            #DataSymbols_nnorm[k, int(l-1)] = 0

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
                #if np.abs(np.real(d)-np.real(DataSymbols_nzero[j]))<0.06 and  np.abs(np.imag(d)-np.imag(DataSymbols_nzero[j]))<0.06:
                #    bit_vect.extend(bits)
                #    break

            cnstl_of_carrier.append(DataSymbols_nzero)
            vector.append(np.array(bit_vect))
        return vector, cnstl_of_carrier
        
    def genNRdownlink(self,BW,BWP,osr,cnstl):
        signal=[]
        REs=[]
        for n in range(0,len(BWP)):
            mu=BWP[n][0]
            dl=self.NRparameters(mu,BW,osr[n])
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
            dl2=self.NRparameters(mu,BW_of_BWP,osr[n])  # To check if BWP is more than 20 RB
            RE[:start,:]=0
            RE[stop:,:]=0
            #RE=np.zeros((dl["Nsc"],Nsymb),complex)
            symb_in_frame=int(2**mu*10*14)
            for i in range(0,int(np.ceil(Nsymb/symb_in_frame))):
                if (Nsymb-i*symb_in_frame)>symb_in_frame:
                    NsymbInGrid=symb_in_frame
                else:
                    NsymbInGrid=(Nsymb-i*symb_in_frame)
                
                r_DMRS=self.genDMRS(N_ID_cell,NsymbInGrid,(stop-start)/12) # DeModulation Reference Signals
                
                # map DMRS
                #v_shift=N_ID_cell%6
           
                k=np.arange(start,stop,2)
            
                for l in range(0,int(NsymbInGrid)):
                    if l%14==2 :

                        RE[k,i*symb_in_frame+l]=r_DMRS[:,l]
               
            # map PSS
            d_PSS=self.genPSS(N_ID_2) # PRIMARY SYNCHRONIZATION SIGNAL
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
            #signal.append(raw_vector)
            #REs.append(RE)
            #raw_vector=np.concatenate(signal)
            # apply BB filter (if activated)
            #if self.fil=="off":
            final_vector=raw_vector
            #else:
            #    final_vector=self.NRfilter(mu,raw_vector,BW,osr[n])
            #    #final_vector=self.RRCfilter(mu,raw_vector,BW,osr[n])

            #s=self.normalize(final_vector,"max",1) # normalize output signal to 1
            s=final_vector
            #s[:]=1
            signal.append(s)
            REs.append(RE)
        out={
        "s":signal,
        "cnstl":REs,
        "Fs":dl["Fs"]
        }

        return out




        
    def normalize(self,x,opt,k):
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

        elif opt=="pow":
            pow=(np.absolute(x)**2).sum()/len(x)
            y=k*x/np.sqrt(pow)

        elif opt=="totpow":
            """TBD if needed"""
        return y

    def NRfilter(self,Fs,raw_vector,BW,*arg):
        # get sampling frequency
        #up=self.NRparameters(mu,BW,osr)
        #Fs=up["Fs"]
        Fs=int(Fs)
        if len(arg)<1:
            osr=1
        else:
            osr=int(arg[0])
        # select filter parameters
        #fac=1
        fac=BW/50e6
        if fac<1 or osr==1:
            fac=1
        order=self.fil_len*np.ceil(osr*fac) # filter order
        #order=70*3
        #dF=600e3*BW/20e6  # half width of transition bandwidth
        dF=0
        for mu in range(0,5):
            try:
                up=self.NRparameters(4-mu,BW)
                if ((up["RB"])*12*15e3*2**(4-mu))>dF:
                    dF=(up["RB"])*12*15e3*2**(4-mu)
            except:
                continue
        dF=(BW-dF)*0.5
        #Fc=(up["RB"]*12*15e3*2**mu)*0.5
        Fc=BW/2 # center of transition bandwidth
        t_vect=np.arange(0,len(raw_vector))/Fs
        # filter NR signal with circular convolution
        #b=sig.remez(int(order+1),[0,BW,(Fs/2-BW),0.5*Fs],[1,0],Hz=int(Fs)) 
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
        #while len(bb)<len(raw_vector):
        #   bb=np.concatenate((bb,b))
        #bb=np.pad(b,(0,len(raw_vector)-len(b)),constant_values=0)
        #s_fil=np.fft.ifft(np.fft.fft(raw_vector,len(raw_vector))*np.fft.fft(bb,len(raw_vector)))
        s_fil=np.convolve(raw_vector.reshape((-1,1))[:,0],bb,mode='full').reshape((-1,1))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        #s_fil=s_fil[0::2,0].reshape((-1,1))
        # compensate group delay (order/2)
        s_out=np.concatenate((s_fil[int(order/2):],s_fil[0:int(order/2)]))
        #s_out=s_fil
        return s_out[:,0]
        
    def RRCfilter(self,mu,raw_vector, BW,osr):
        # get sampling frequency
        up=self.NRparameters(mu,BW,osr)
        Fs=up["Fs"]
        # select filter parameters
        order=500*osr # filter order
        N=order+1
        alpha=0.5

        """
        Generates a root raised cosine (RRC) filter (FIR) impulse response.
        Parameters
        ----------
        N : int
            Length of the filter in samples.
        alpha : float
            Roll off factor (Valid values are [0, 1]).
        Ts : float
            Symbol period in seconds.
        Fs : float
            Sampling Rate in Hz.
        Returns
        ---------
        time_idx : 1-D ndarray of floats
            Array containing the time indices, in seconds, for
            the impulse response.
        h_rrc : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.
        """


        T_delta = 1/float(Fs)
        time_idx = ((np.arange(N)-N/2))*T_delta
        sample_num = np.arange(N, dtype=int)
        h_rrc = np.zeros(int(N), dtype=float)
        Ts=1
        #Ts=(0.5/7*10e-3)
        for x in sample_num:
            #t=(x-N/2)
            t = (x-N/2)*T_delta
            if t == 0.0:
                h_rrc[x] = (1.0 - alpha + (4*alpha/np.pi))
            elif alpha != 0 and (t == Ts/(4*alpha) or  t == -Ts/(4*alpha)) :
                h_rrc[x] = ((alpha/np.sqrt(2))*(((1+2/np.pi)* \
                        (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha))))))
            else:
                h_rrc[x] =((np.sin(np.pi*t*(1-alpha)) +  \
                        4*alpha*(t)*np.cos(np.pi*t*(1+alpha)))/ \
                        (np.pi*t*(1-(4*alpha*t)*(4*alpha*t))))
        h_rrc=h_rrc/np.sqrt(sum(h_rrc**2))
        h_rrc=np.fft.fft(h_rrc)
        #bb=np.pad(h_rrc,(0,len(raw_vector)-len(h_rrc)),constant_values=0)
        bb=h_rrc
        # Different implementation of same filter

        Tsample =1/Fs #(0.5/7*10e-3)/up["NFFT"] # sampling period, should at least twice the rate of the symbol

        
        Tsymbol = osr * Tsample # pulse duration should be at least 2 * Ts
        span = 70 # number of symbols to span, must be even
        n = int(span*osr) # length of the filter = samples per symbol * symbol span

        # t_step must be from -span/2 to +span/2 symbols.
        # each symbol has "sps" number of samples per second. 
        #t_step = Tsample * np.linspace(-n/2,n/2,n+1) # n+1 to include 0 time
            #t_step = np.linspace(-n/2,n/2,n+1)/osr
            #BW = (1 + alpha) / Tsymbol
            #a = np.zeros_like(t_step)

            #for item in list(enumerate(t_step)):
            #   i,t = item
            #   # t is n*Ts
            #   if np.abs(np.abs(4.0*alpha*t)-1) < np.sqrt(10e-16):
            #       a[i] = 1/(2*np.pi*osr)*(np.pi*(alpha+1)*np.sin(np.pi*(alpha+1)/(4*alpha))-4*alpha*np.sin(np.pi*(alpha-1)/(4*alpha))+np.pi*(alpha-1)*np.cos(np.pi*(alpha-1)/(4*alpha)))
            #   elif t == 0:
            #       a[i] = -1/(np.pi*osr)*(np.pi*(alpha-1)-4*alpha)

            #   else:
            #       a[i] =  -4*alpha/osr*(np.cos((1+alpha)*np.pi*t)+np.sin((1-alpha)*np.pi*t)/(4*alpha*t))/(np.pi*((4*alpha*t)**2-1))

            #a=a/np.sqrt(sum(a**2))
            #bb=a
        #for item in list(enumerate(t_step)):
        #   i,t = item
        #   # t is n*Ts
        #   if (1-(2.0*alpha*t/Tsymbol)**2) == 0:
        #       a[i] = np.pi/4 * np.sinc(t/Tsymbol)
        #   elif t == 0:
        #       a[i] = np.cos(alpha * np.pi * t / Tsymbol)/ (1-(2.0*alpha*t/Tsymbol)**2)
            

        #   else:
        #       numerator = np.sinc( np.pi * t/Tsymbol )*np.cos( np.pi*alpha*t/Tsymbol )
        #       denominator = (1.0 - (2.0*alpha*t/Tsymbol)**2)
        #       a[i] =  numerator / denominator
        
        #a=a/np.sqrt(sum(a**2))
        #bb=np.pad(a,(0,len(raw_vector)-len(a)),constant_values=0)



        #s_fil=np.fft.ifft(np.fft.fft(raw_vector,len(raw_vector))*np.fft.fft(bb,len(raw_vector)))
        s_fil=np.convolve(raw_vector,bb[0:len(raw_vector)],mode='same') 
        
        # compensate group delay (order/2)
        s_out=np.concatenate((s_fil[int(order/2):],s_fil[0:int(order/2)]))
        
        return s_out


    
    def demNRdownlink(self,s,BW,BWP,osr):
        
        FFTwpos=0
        end_of_prev_sig=0
        equalize=self.equalizer
        cnstl_of_carrier=[]
        #if self.fil=="on":
        #    sign=self.RRCfilter(mu,sign,BW,osr[n])
        #    #sign=self.NRfilter(sign,BW,osr)
        for n in range(0,len(BWP)):
            mu=BWP[n][0]
            N_symb_TOT=int(BWP[n][1])
            Nsym_cp1=np.ceil(N_symb_TOT/(7*2**mu)) 

            
            # get various parameters related to input signal
            dl=self.NRparameters(mu,BW,osr[n]) # general NR parameters
            Nsym_cp2 = N_symb_TOT - Nsym_cp1
            N_sampl = int(dl["Nofdm1"]*Nsym_cp1 + dl["Nofdm2"]*Nsym_cp2) # get total number of samples
            sign=s[end_of_prev_sig:end_of_prev_sig+N_sampl]
            end_of_prev_sig=end_of_prev_sig+N_sampl
            #if self.fil=="on":
            #    #sign=self.RRCfilter(mu,sign,BW,osr[n])
            #    sign=self.NRfilter(mu,sign,BW,osr[n])
            #N_sampl=len(sign) # total number of signal samples
            N_slots=np.floor(N_symb_TOT/14) # number of slots (integer)
            #rem_sampl=N_sampl-N_slots*dl["Nslot"] # remaining samples at signal tail
            #N_symb=np.floor(rem_sampl/dl["Nofdm1"]) # number of OFDM symbols at signal tail
            #if N_symb >0: # if at least one OFDM symbol, check for the next ones
            #    N_symb=1+np.floor((rem_sampl-dl["Nofdm1"])/dl["Nofdm2"])
            #N_symb_TOT=14*N_slots+N_symb # total number of OFDM symbols (whole signal)

            #N_RB_maxDL=110 # largest downlink bandwidth (in Resource Blocks)
            N_ID_1=0    # physical-layer cell-identity group
            N_ID_2=0    # physical-layer identity within the group
            N_ID_cell=3*N_ID_1+N_ID_2 # cell identity
            # generate PSS in the time-domain
            d_PSS=self.genPSS(N_ID_2) # generate PSS sequence
            # map PSS sequence to subcarriers
            PSSf=np.zeros((int(dl["NFFT"])),complex)
            PSSf[-64:]=d_PSS[0:64]
            PSSf[1:64]=d_PSS[64:127]
            PSSt=np.fft.ifft(PSSf) # convert to time-domain
            # find match of PSS within the input vector, by computing cross-correlation
            # and looking for its maximum
            pad=np.pad(PSSt, (0,len(sign)-len(PSSt)), constant_values=0)
            xcmax=np.argmax(np.absolute(np.correlate(sign,pad,"full"))) 
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
            #invect_aligned[:]=1
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
                # calculate slot and symbol indexes of current OFDM symbol
                #slot=np.floor(index/7)
                #symbol=index % 7
                ## calculate start & end indexes of FFT window
                #if symbol==0:
                #    i1 = slot*dl["Nslot"] + dl["Ncp1"] + FFTw_shift_1 + 1
                #    i2 = slot*dl["Nslot"] + dl["Ncp1"] + FFTw_shift_1 + dl["NFFT"]
                #else:
                #    i1 = slot*dl["Nslot"] + dl["Nofdm1"] + (symbol-1)*dl["Nofdm2"] + dl["Ncp2"] + FFTw_shift_2 + 1
                #    i2 = slot*dl["Nslot"] + dl["Nofdm1"] + (symbol-1)*dl["Nofdm2"] + dl["Ncp2"] + FFTw_shift_2 + dl["NFFT"]
                # copy FFT window to symbol matrix
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
                
                r_DMRS=self.genDMRS(N_ID_cell,NsymbInGrid,(stop-start)/12) # DeModulation Reference Signals
                
                # map DMRS
                #v_shift=N_ID_cell%6
           
                k=np.arange(start,stop,2)
            
                for l in range(0,int(NsymbInGrid)):
                    if l%14==2 :

                        RE_id[k,i*symb_in_frame+l]=r_DMRS[:,l]
               
            #r_DMRS = self.genDMRS( N_ID_cell,int(N_symb_TOT));
            #RE_id = np.ones((RE.shape),complex);
            #v_shift = N_ID_cell% 6
            #m = np.arange(0,int(2*dl["RB"]))
            #mp = m + N_RB_maxDL - dl["RB"]

            #for l in range(0,int(N_symb_TOT)):
            #    if l% 7 == 0 or l% 7 == 4:
            #        v = 3 * ((l% 7) !=0)
            #        k = 6*m + (v+v_shift)% 6
            #        RE_id[k,l] = r_DMRS[mp,l]
            d_PSS=self.genPSS(N_ID_2) # PRIMARY SYNCHRONIZATION SIGNAL
            #if n==0:
            l=min([NsymbInGrid,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
            k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
            first=int(k[0])
            last=int(k[-1])
            RE_id[first:last+1,i*symb_in_frame+l]=np.concatenate((np.zeros(56),d_PSS,np.zeros(57))) # set  edge subcarriers to 0

            #l = min(N_symb_TOT, 7) # if signal shorter that 1 slot --> anticipate PSS!
            #k_PSS = np.arange(0,72) - 31 + int(dl["Nsc"]/2) - 5
            #RE_id[k_PSS,int(l-1)] = np.concatenate((np.zeros(5),d_PSS,np.zeros(5))) # set 5+5 edge subcarriers to 0

            # calculate the complex ratios of the post-FFT acquired signal "RE" and the
            # post-FFT ideal signal "RE_id", for each reference symbol
            complex_ratios = np.divide(RE, RE_id)
            #test=np.argwhere(RE_id==0)
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



            #l = np.arange(0,N_symb_TOT,7,dtype=int) 
            #for k in (6*m + v_shift):
            #    for i in  np.arange(1,l.size):
            #        delta_phi = phi[k,int(l[i])] - phi[k,int(l[i-1])]
            #        if np.absolute(delta_phi) >= np.pi:
            #            phi[k,l[i:]] = phi[k,l[i:]] - 2*np.pi*np.sign(delta_phi)
            ## unwrap phase of complex ratios at symbol #4 of each slot
            #l = np.arange(4,N_symb_TOT,7,dtype=int)
            #for k in (6*m + (3+v_shift)%6):
            #    for i in np.arange(1,l.size):
            #        delta_phi = phi[k,l[i]] - phi[k,l[i-1]]
            #        if np.absolute(delta_phi) >= np.pi:
            #            phi[k,l[i:]] = phi[k,l[i:]] - 2*np.pi*np.sign(delta_phi)

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

            #l = np.arange(0,N_symb_TOT,7,dtype=int)

            #for k in (6*m + v_shift):
            #    test=a[int(k),l]
            #    a_avg[int(k)] = np.mean(a[int(k),l])
            #    phi_avg[int(k)] = np.mean(phi[int(k),l])
            #l = np.arange(4,N_symb_TOT,7,dtype=int)
            #for k in (6*m + (3+v_shift)%6):
            #    a_avg[int(k)] = np.mean(a[int(k),l])
            #    phi_avg[int(k)] = np.mean(phi[int(k),l])

            # the equalizer coefficients for amplitude and phase "a_coeff" and
            # "phi_coeff" at the reference signal subcarriers are obtained by computing
            # the moving average in the frequency domain of the time-averaged reference
            # signal subcarriers, i.e. every third subcarrier (or sixth, if less than 5
            # OFDM symbols)
            a_coeff =np.zeros((int(dl["Nsc"])))
            phi_coeff = np.zeros((int(dl["Nsc"])))
            k_PSS=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
            k=np.arange(start,stop,2)
            #if N_symb_TOT < 5 : # no CRS transmitted on subcarriers 3, 9, 15, ...
            #    k = 6*m + v_shift
            #else:
            #    m = np.arange(0,4*dl["RB"])
            #    k = 3*m + v_shift
        
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
        
            #test= a_coeff[k]
            #test1=(k+1)
            # perform linear interpolation to compute coefficients for each subcarrier
            a_coeff = np.interp(np.arange(1,dl["Nsc"]+1), k+1, a_coeff[k] )
            phi_coeff = np.interp(np.arange(1,dl["Nsc"]+1), k+1, phi_coeff[k] )

            cnstl = np.zeros((RE.shape),complex)
            # equalize resource elements and return
            for i in np.arange(0,dl["Nsc"]):
                cnstl[i,:]=RE[i,:]/(a_coeff[i]*np.exp(1j*phi_coeff[i]))

            cnstl_of_carrier.append(cnstl)


        return cnstl_of_carrier




    def measEVMdownlink(self, BW, BWP, cnstl, dem,*arg):

        

        Nsc,N_symb=cnstl.shape # get some info from constellation matrix dimensions
        ## determine system bandwidth from number of subcarriers
        #if Nsc==72:
        #     BW = 1.4e6;
        #elif  Nsc==180:
        #    BW = 3e6
        #elif  Nsc== 300:
        #    BW = 5e6
        #elif  Nsc== 600:
        #    BW = 10e6
        #elif  Nsc== 900:
        #    BW = 15e6
        #elif  Nsc== 1200:
        #    BW = 20e6
        #else:
        #    return 0,0
        mu=BWP[0]
        if dem==[]:
            return 0,0
        Nsc_rx, N_symb_rx = dem.shape
        if Nsc!=Nsc_rx or   N_symb!=N_symb_rx:
            return 0,0
        dl=self.NRparameters(mu,BW) # get NR parameters
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
        #v_shift=N_ID_cell%6
        #m=np.arange(0,2*dl["RB"])
        #for l in np.arange(0,N_symb):
        #    if l%7==0 or l%7==4:
        #        v=3*((l%7)!=0)
        #        k=6*m+(v+v_shift)%6
        #        rxDataSymbols_nnorm[k,l] = 0
        #        refDataSymbols_nnorm[k,l] = 0
        
        # set PSS symbols to zero
        l=min([N_symb,4])-1 # if signal shorter that 4 symbols --> anticipate PSS!
        k=np.arange(0,240)- np.ceil(240/2) + dl["Nsc"]/2
        first=int(k[0])
        last=int(k[-1])
        rxDataSymbols_nnorm[first:last+1,l]=0
        refDataSymbols_nnorm[first:last+1,l]=0
        #l = min(N_symb, 7)  
        #k= np.arange(0,72) - 31 + int(dl["Nsc"]/2) - 5
        #rxDataSymbols_nnorm[k, int(l-1)] = 0
        #refDataSymbols_nnorm[k, int(l-1)] = 0
        # set non-equalized symbols to zero
        #if N_symb < 5:
        #    rxDataSymbols_nnorm[-5:,:] = 0
        #    refDataSymbols_nnorm[-5:,:] = 0
            
        #else:
        #    rxDataSymbols_nnorm[-2:,:] = 0
        #    refDataSymbols_nnorm[-2:,:] = 0

        # vectorize constellations
        rxDataSymbols_vect = np.reshape(rxDataSymbols_nnorm,(rxDataSymbols_nnorm.size),order="F" );
        refDataSymbols_vect = np.reshape(refDataSymbols_nnorm,(refDataSymbols_nnorm.size),order="F" );

        #test=np.argwhere(refDataSymbols_vect)
        # keep only non-zero symbols
        #for i in range(0,len(rxDataSymbols_vect)):
        #   if np.abs(rxDataSymbols_vect[i])<10e-12:
        #       rxDataSymbols_vect[i]=0
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
        #for i in range(0,len(received)-len(reference)):
        #    reference=np.concatenate((reference, np.array([[0]],complex)))
        
        # Do not take zeros into account
        #received=np.delete(received,range(len(reference),len(received)))


        #Shape the vectors: time is row observation is column
        #if received.shape[0]<received.shape[1]:
        #    received=np.transpose(received)
        reference.shape=(-1,1)
        received.shape=(-1,1)

        #RMS for Scaling
        rmsref=np.std(reference)
        rmsreceived=np.std(received)
        #EVM=(np.mean(np.mean(np.abs(received-reference)**2,axis=0)/np.mean(np.abs(reference)**2,axis=0)))**(1/2)
        EVM=(np.mean(np.mean(np.abs(received-reference)**2,axis=0)/np.mean(np.abs(reference)**2,axis=0)))**(1/2)
        #EVM=10*np.log10(np.mean(np.mean(np.abs(received/rmsreceived*rmsref-reference)**2,axis=0)/np.mean(np.abs(reference)**2,axis=0)))
# Copied part ends
        
        return EVM, rxDataSymbols
        


        





    def define_io_conditions(self):
        
        # Input A is read to verilog simulation after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['A']._io_condition='initdone'
        # Output is read to verilog simulation when all of the outputs are valid, 
        # and after 'initdone' is set to 1 by controller
        self.iofile_bundle.Members['Z'].verilog_io_condition_append(cond='&& initdone')

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
    

    #s=x.RRCfilter(s,BW,x.osr)
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
    fig=plt.figure()
    plt.plot(f_plot/(10**6),y_plot)
    plt.grid()
    plt.title("Signal spectrum")
    plt.show()
    return fig


def plot_PSD2(x,a,*arg):
    if len(arg)>=1:
        Fs=int(arg[0])
        s=x
        
    else:
        Fs = int(x.s_struct["Fs"])
        s=x
        
    

    #s=x.RRCfilter(s,BW,x.osr)
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
    plt.show(block=False)


if __name__=="__main__":
    #import matplotlib.pyplot as plt
    #from  NR_signal_studio import *
    #from  NR_signal_studio.controller import controller as NR_signal_studio_controller

    length=1024
    #method="multi"
    #method="single"
    rs=100e6
    #BW=np.ones(5)*1.4e6  # vector of carrier bandwidths (1.4, 3, 5, 10, 15, or 20 MHz)
    #BW=np.array([3.6e6,20e6])
    #BWP=np.array([[[0,14,0,1],[0,14,0,1]],[[0,14,0,1],[1,14,0,1],[2,14,0,1]]])  #[mu, symbols, BW_low(0...1),BW_high(0...1)>BW_low]
    BWP=np.array([[[1,14*2,0,1]],[],[[1,14*2,0,1]]])
    BW=np.array([10e6,-10e6,10e6])
    mu=[0,0]
    #BW=10e6
    #QAM="16QAM"
    QAM="64QAM"
    #QAM="256QAM"
    #QAM="4QAM"
    #QAM="BPSK" # Not for downlink
    M=16
    nsc=72
    k=int(np.log2(M))
    #bits=np.array([0,0,0,1]*3816)

    #bits=np.array(list(bits)*(int(240/4))) # same as ones in matlab nsymb 2 16qam
    #bits=np.array(list(bits)*(int(6000)))
    #bits=np.random.randint(2,size=1028*2) #Naymb=2
    bits=np.random.randint(2,size=3810*6)  #Nsymb=7
    bits2=np.random.randint(2,size=47520*6)
    bits3=np.random.randint(2,size=7536*6)
    #bits=np.random.randint(2,size=1710*4) 
    multi_bits=[bits]*len(BW)
    multi_bits =np.array([[bits3],[],[bits3]])
    #multi_bits[0]=[]
    osr=1
    #Nsymb=10  
    testvar=0
    if not hasattr(BW,"__len__") :
        a=5
        #test=NR(BW,BWP,osr,Nsymb,QAM,bits)
        #print(test.EVM)
        #plt.figure()
        #plt.plot(test.rxDataSymbols.real,test.rxDataSymbols.imag,'o')
        #plt.show(block=False)
        #a=plot_PSD(test,100)
        #plt.show()
    elif  hasattr(BW,"__len__") :
        test=NR(BW,BWP,osr,QAM,multi_bits)
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
    
    meas(test,test.s_struct["Fs"])
    a=plot_PSD(test,100,test.s_struct["Fs"])
    print("tst")
