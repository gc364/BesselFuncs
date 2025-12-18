import scipy.special as spec
import matplotlib.pyplot as plt
import torch
import BesselFuncs.BesselFuncs as BesselFuncs

def tests():
    ###############################
    #########BSecondOrder Tests####
    ###############################
    ####Bessel function of the#### 
    ####    Second Kind       ####
    x = torch.linspace(0.5,100,1000)
    eps = torch.tensor(1e-15)
    fig, ax  = plt.subplots()
    for nu in [0,0.2,0.4]:
        a =torch.tensor(nu)
        ya,ya1 = BesselFuncs.BSecondOrder(a,x,eps,a,a)
        ax.plot(x,ya,label=f'{nu}')
        ax.plot(x,spec.yv(nu,x),linestyle='--')
    ax.set_ylim(-1)
    plt.legend()
    #plt.show()
    plt.close()
    ######################
    ######ZSERI TEST#####
    #####################
    fig, ax  = plt.subplots()
    x = torch.linspace(-2,2,100)#[-2,-1,0,1,2]
    I = [-2j,-1j,0,1j,2j]
    C = [1-2j,1-1j,0,1+1j,1+2j]
    args = [x,torch.tensor(I),torch.tensor(C)]
    labels = ['Real','Imaginary','Complex']
  
    for nu in [0,0.5,1,2]:
        print('#'*10)
        print(f'Order = {nu} \n')
        for i in range(3):
            a =torch.tensor(nu)
            ya = BesselFuncs.ZSERI(a,args[i])
            real = spec.iv(nu,args[i]) ###This looks wrong, it keeps returing nans
           
            if i==0 :
                ax.plot(args[i],ya,label=f'{nu}')
                ax.plot(args[i],real,linestyle='--',label=f'{nu}')
                ax.set_title(labels[i])
                print(f'{labels[i]} % Error: {(((spec.iv(nu,args[i])-ya)/ya).mean())*100} %')

            else:
                print(f'{labels[i]} % Error: {(((spec.iv(nu,args[i])-ya)/ya).mean())*100} %')
    plt.legend()
    #plt.show()
    plt.close()
    ######################
    ######ZASYI TEST#####
    #####################
    print('#'*20)
    print(f'{'#'*5} ZASYI Tests {'#'*2}')
    print('#'*20)
    fig, ax  = plt.subplots()
    x = torch.linspace(5,10,100)#[-2,-1,0,1,2]
    I = [0.1j,1j,2j]
    C = [0.1,1+1j,1+2j,2+1j,2+2j]
    args = [x,torch.tensor(I),torch.tensor(C)]
    labels = ['Real','Imaginary','Complex']
  
    for nu in [-.1,0,0.5,1,2]:
        print('#'*10)
        print(f'Order = {nu} \n')
        
        for i in range(3):
            
            a =torch.tensor(nu)
            ya = BesselFuncs.ZASYI(a,args[i])#*torch.exp(torch.tensor(1j*torch.pi/2))
            
            if i==0 :
                ax.plot(args[i],ya.real,label=f'{nu}')
                ax.plot(args[i],spec.iv(nu,args[i]).real,linestyle='--',label=f'{nu}')
                ax.set_title(labels[i])
                ax.set_ylim(-2,2)
                print(f'{labels[i]} % Error: {(((ya-spec.iv(nu,args[i]))/ya).mean())*100} %')

            else:
                print(f'{labels[i]} % Error: {(((spec.iv(nu,args[i])-ya)/ya).mean())*100} %')
    plt.legend()
    #plt.show()
    plt.close()

    ##################
    ######jv TEST#####
    ##################
    print('#'*20)
    print(f'{'#'*5} jv Tests {'#'*2}')
    print('#'*20)
    fig, ax  = plt.subplots()
    x = torch.linspace(-10,10,100,dtype=torch.complex128)#[-2,-1,0,1,2]
    I = [0.1j,1j,2j]
    C = [0.1,1+1j,1+2j,2+1j,2+2j]
    args = [x,torch.tensor(I),torch.tensor(C)]
    labels = ['Real','Imaginary','Complex']
  
    for nu in [-2,-1,-0.5,0,1]:
        print('#'*10)
        print(f'Order = {nu} \n')
        
        for i in range(3):
            
            a =torch.tensor(nu)
            ya = BesselFuncs.jv(a,args[i])
            
            if i==0 :
                ax.plot(args[i],ya.real,label=f'{nu}')
                ax.plot(args[i],spec.jv(nu,args[i]).real,linestyle='--',label=f'{nu}')
                ax.set_title(labels[i])
                print(f'{labels[i]} % Error: {(((ya-spec.jv(nu,args[i]))/ya).mean())*100} %')

            else:
                print(f'{labels[i]} % Error: {(((spec.jv(nu,args[i])-ya)/ya).mean())*100} %')
    ax.set_ylim(5,-5)
    plt.legend()
    #plt.show()
    plt.close()
    #### x hould have dim(nu) columns and z rows.


    #########THIS IS EXACTLY WHAT PYPROP8 DOES, MOST IMPORTANT TEST##########
    #########################################################################
    #########################################################################
    x = torch.linspace(-10,10,100,dtype=torch.complex128).tile([5,1]).T
    print(x.shape)
    nu = torch.arange(-2,3)
    fig,ax = plt.subplots()
    J = BesselFuncs.jv(nu,x)
    Jr  = spec.jv(nu,x)
    for i in range(5):
        ax.plot(x[:,i].real,J[:,i].real,label=f'{nu[i]}')
        ax.plot(x[:,i].real,Jr[:,i].real,label=f'{nu[i]}',linestyle='--')
    ax.set_ylim(5,-5)
    plt.legend()
    plt.show()    
    plt.close()
    ###########################
    ###########jvp Test########
    print('#'*20)
    print(f'{'#'*5} jvp Tests {'#'*2}')
    print('#'*20)
    fig,ax = plt.subplots()
    jvpr = spec.jvp(nu,x,1)
    jvp = BesselFuncs.jvp(nu,x,1)
    for i in range(5):
        ax.plot(x[:,i].real,jvp[:,i].real,label=f'{nu[i]}')
        ax.plot(x[:,i].real,jvpr[:,i].real,label=f'{nu[i]}',linestyle='--')
    plt.legend()
    plt.show()  
    ax.set_ylim(5,-5)  
    print(f'% Error: {((jvpr-jvp)/jvpr).mean()*100} %')

    ##########Backprop test######################
    print('#'*20)
    print(f'{'#'*5} Backprop Tests {'#'*2}')
    print('#'*20)
    ## x is a leaf node 
    x = torch.linspace(-10,10,100,requires_grad=True)
    j = BesselFuncs.jv(torch.tensor(0),x)
    j.real.mean().backward()
    print(f'dj/dx = {x.grad}')



tests()