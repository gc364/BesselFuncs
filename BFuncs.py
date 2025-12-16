import torch
import scipy.special as spec
import matplotlib.pyplot as plt
import math


"""To recreate `jv` (Bessel function of the first kind of real order and complex argument), we need to recreate 
`zbesj.f`, from AMOS for positive order and `zbesy.f` for negative order (Already done). 
`zbesj.f` uses the SLATEC function ZBINU. This uses :
[ZABS (abs of complex number), 
ZASYI (Modified Bessel for real argument with asymptotic expansion),
ZBUNI (Modified Bessel for large abs(Z)), 
ZMLRI (Modified Bessel for real argument with Miller algorithm), 
ZSERI (Modified Bessel for real argument with Power series expansiom), 
ZUOIK (Finds underflows in asymptotic expansions), 
ZWRSK (Modified Bessel for real argument by normalising ratios from ZRATI),
ZRATI (Computes ratios of modified Bessels by backward recurrence )
] 
from SLATEC.
Essentially ZBINU is a main function for the routines above, and chooses which
to use based off the argument. To save time we will only recreate the ones important
for pyprop8's usecase.

UPDATE: We can calulate for complex z and real nu, modified Bessel functions of the first kind where -2<abs(z)<inf
        Now we can try and get jv working for this limited range

Order range  = [-2,3]

 """


class BesselFuncs:
    def __init__(self):
        pass
    
    def Pochhammer(self,nu,k):
        """Also called a rising factorial power"""
        if k==0:
            return 1
        num=1
        s = 0
        for i in range(1,2*k,2):
            
            num*=((4*nu**2)-(i**2))
            s+=1/((4*nu**2)-(i**2))
        den = 2**(3*k)*torch.special.gammaln(torch.tensor(k+1)).exp()
        
        # norm = ((-1)**k)/((2**k)*torch.special.gammaln(torch.tensor(k)).exp())
        
        # num = torch.special.gammaln(.5+nu+k).exp()
        # den  = torch.special.gammaln(.5+nu).exp()
        # t1 = num/den
        
        # num = torch.special.gammaln(.5-nu+k).exp()
        # den  = torch.special.gammaln(.5-nu).exp()
        #t2 = num/den
        return num/den
        


    def ZASYI(self,nu,z,BR=False):
        """Modified Bessel function 1st kind using the asymptotic expansion for large abs(z)
        Only valid for positve abs(z), best with abs(nu)<1"""
        N = math.ceil(max(abs(nu)-0.5,1))+10
       
        norm2 = torch.exp(z)/torch.sqrt(2*torch.pi*z)
        norm1 = (1j*torch.exp(-z+torch.pi*nu*1j))/torch.sqrt(2*torch.pi*z)
        if BR:
            norm1 = (1j*torch.exp(-z-torch.pi*nu*1j))/torch.sqrt(2*torch.pi*z)
        t1=0
        t2=0
        for k in range(N):
            a = self.Pochhammer(nu,k)
            
            #Term 1
            t1+=a/((z)**k)
            #Term 2
            t2+= ((-1)**(k)*a)/((z)**k)

        return norm1*t1+norm2*t2
        
        
        # norm  = ((z**nu)*(-z**2)**(-(2*nu+1)/(4)))/math.sqrt(torch.pi*2)
        # print(norm)
        # s = 0
        # for k in range(N):
        #     s+=(1/(z**k))

        # t1 = torch.exp(-1j*(((2*nu+1)*torch.pi)/(4)-torch.sqrt(-z**2)))*s
        # t2 = torch.exp(1j*(((2*nu+1)*torch.pi)/(4)-torch.sqrt(-z**2)))*s
        # print(t1)
        # print(t2)
        # return norm*(t1+t2)
        #return (1j*torch.exp(1j*nu*torch.pi))*norm1*t1+norm2*t2

    
    def ZSERI(self,nu,z):
        """Power series expansion for calculating modified bessel functions 1st kind with complex argument
        Valid for abs(z) = [-2,2]"""
        #c= (z/2)**nu
        s = 0
        for k in range(100):
            num = (z/2)**(2*k+nu)
            denom = torch.special.gammaln(torch.tensor(k+1)).exp()*torch.special.gammaln(nu+k+1).exp()
            s+=num/denom
        return s


    def jv(self,nu,x,eps=1e-12,ya=0,ya1=0):
        """Reimplementation of scipy jv using pytorch.
        Follows AMOS FORTRAN implementation zbesj.f
        Bessel function of the first kind"""
        #x+=1e-6j
        J = torch.zeros_like(x)
        if torch.sign(nu) != -1.:
            for zi in range(x.shape[0]):
                ###For imag(Z)>0
                if x[zi].imag > 0:
                    I= self.ZBINU(nu,-1j*x[zi])
                  
                    J[zi] = torch.exp(nu*torch.pi*1j/2)*I
                else:
                    ##For Imag(Z)< 0
                    I= self.ZBINU(nu,1j*x[zi])
                    J[zi] = torch.exp(-nu*torch.pi*1j/2)*I
            return J

        
    def ZBINU(self,nu,z):
        """Conglomeration of all the methods of computing modified bessel functions of the 1st kind.
        Will choose the appropiate one given the values of z"""
       
        if (torch.sign(torch.min(z.real)) == -1 ) :
            if (torch.abs(z)<=5) :
                return self.ZSERI(nu,z)
            else:
                rCp =  self.ZASYI(nu,torch.exp(1j*torch.pi)*z)#Calculate in the right half of the complex plane
                print(rCp)
                return rCp *torch.exp(-nu*torch.pi)##spin around to the left
        else:
            if (torch.abs(z)<=5) :
                return self.ZSERI(nu,z)
            elif z.imag <0:
                return self.ZASYI(nu,z*torch.exp(torch.tensor(1j*torch.pi)))
            else:
                return self.ZASYI(nu,z)
        


    def BSecondOrder(self,a,xf,eps,ya,ya1):
        """Reimplemtation of Bessel functions of the second kind, from Temme 1976
        a: Order: Tensor (Real)
        x: Evaluation Points (Real)
        eps: Precision 
        ** Return buffers
        
        Note:
        Can take non-integer orders"""
        p=q=0
        h=0
        
        pi = 4*torch.arctan(torch.tensor(1))
        na = int(a+0.5)
        rec = a >=0.5
        rev =a<-0.5
        if rev or rec:
            a = a-na
        ya1f =torch.zeros_like(xf)
        yaf= torch.zeros_like(xf)
        for comp in range(xf.shape[0]):
            x = xf[comp]
            if a == -.5:
                p=torch.sqrt(2/pi/x)
                f = p*torch.sin(x)
                g= -p*torch.cos(x)
            elif x<3:
                b = x/2
                d=-torch.log(b)
                e = a*d

                c = 1/pi if abs(a)<1e-15 else a/torch.sin(a*pi)
                s = 1 if abs(e)<1e-15 else self.sinh(e)/e
                e = torch.exp(e)
                
                recip,p,q = self.reiprocal_gamma(a,p,q)
                g = e*recip
                e = (e+1/e)/2
                f=2*c*(p*e+q*s*d)
                e=a**2
                p=g*c
                q=1/g/pi
                c = a*pi/2
                r = 1 if abs(c)<1e-15 else torch.sin(c)/c
                r = pi*c*r*r
                c=1
                d=-b*b
                ya = f+r*q
                ya1=p
                n=1
                while abs(g/(1+abs(ya)))+abs(h/(1+abs(ya1)))>eps:
                    f = (f*n+p+q)/(n*n-e)
                    c = c*d/n
                    p = p/(n-a)
                    q = q/(n+a)
                    g = c*(f+r*q)
                    h = c*p-n*g
                    ya =ya+g
                    ya1 = ya1+h
                    n+=1
                f= -ya
                g= -ya1/b
            else:
                b = x-pi*(a+.5)/2
                c = torch.cos(b)
                s = torch.sin(b)
                d = torch.sqrt(2/x/pi)
                p,q,b,h = self.besspqa(a,x,eps,p,q,b,h)
                f = d*(p*s+q*c)
                g = d*(h*s-b*c)
            if rev:
                x= 2/x
                na = -na-1
                for n in range(0,na):
                    h = x*(a-n)*f-g
                    g=f
                    f=h
            elif rec:
                x= 2/x
                for n in range(1,na):
                    h = x*(a+n)*g-f
                    f=g
                    g=h
          
            yaf[comp]=f.item()
            ya1f[comp]=g.item()
           
        
        return yaf,ya1f
    def sinh(self,x):
        ax = abs(x)
        if ax<.3:
            y = x**2 if ax<.1 else x*x/9
            x = (((1/5040*y +1/120)*y+1/6)*y+1)*x
            sinh = x if ax<.1 else x*(1+4*x*x/27)
        else:
            ax = torch.exp(ax)
            sinh = torch.sign(x)*.5*(ax-1/ax)
        return sinh
    def reiprocal_gamma(self,x,odd,even):
       
        
        b = torch.zeros(12,dtype=torch.float64)
        b[0]    = -.283876542276024
        b[1]    = -.076852840844786
        b[2]    = .001706305071096
        b[3]    = .001271927136655
        b[4]    = .000076309597586
        b[5]    = -.000004971736704
        b[6]    = -.000000865920800
        b[7]    = -.000000033126120
        b[8]    = .000000001745136
        b[9]    = .000000000242310
        b[10]   = .000000000009161
        b[11]   = -.000000000000170

        x2 = x*x*8
        alpha = -.000000000000001
        beta = 0
        for i in range(11,-1,-2):
            
            beta = -(alpha*2 + beta)
            alpha = -beta*x2-alpha+b[i]
        even = (beta/2 + alpha)*x2 -alpha + .921870293650453
        alpha = -.000000000000034
        beta = 0
        for i in range(10,-2,-2):
            
            beta = -(alpha*2 + beta)
            alpha = -beta * x2-alpha+b[i]
        odd = (alpha+beta)*2
        return odd*x+even,odd,even

    def besspqa(self,a,x,eps,pa,qa,pa1,qa1):
        """
        a: order of Bessel (Real)
        x: Evaluation point (Real)
        eps: Precision constant
        
        Remaining are initialised variables to be writtin into and returned
        From Temme 1976. Returns P(a,x),Q(a,x),P(a+1,x) and Q(a+1,x).
        These are used for calculation of ordinary Bessel functions of the second
        kind for |x|>3"""
        rev = a<-.5
        if rev:
            a = -a-1
        rec= a>=.5
        if rec:
            na = int(a+.5)
            a = a-na
        if a==-.5:
            pa = pa1 =1,
            qa = qa1 = 0
        else:
            c=.25-a*a
            b=x+x
            p=4*torch.arctan(torch.tensor(1))
            e = (x*torch.cos(a*p)/p/eps)**2
            p=1
            q=-x
            r=s=1+x*x
            n=2
            while r*n*n<e:
                d=(n-1+c/n)/s
                p=(2*n-p*d)/(n+1)
                q=(-b+q*d)/(n+1)
                s = p*p+q*q
                r=r*s
                n+=1
            f=p=p/s
            g=q=-q/s
            while n>0:
                r=(n+1)*(2-p)-2
                s=b+(n+1)*q
                d=(n-1+c/n)/(r*r+s*s)
                p=d*r
                q=d*s
                e=f
                f=p*(e+1)-g*q
                g=q*(e+1)+p*g                
                n-=1
            f=1+f
            d=f*f+g*g
            pa=f/d
            qa=-g/d
            d=a+.5-p
            q=q+x
            pa1=(pa*q-qa*d)/x
            qa1=(qa*q+pa*d)/x
        if rec:
            x=2/x
            b=(a+1)*x
            for n in range(1,na):
                p0=pa-qa1*b
                q0=qa+pa1*b
                pa = pa1,
                pa1=p0
                qa=qa1
                qa1=q0
                b=b+x
        if rev:
            p0=pa1
            pa1=pa
            pa=p0
            q0=qa1
            qa1=qa
            qa=q0

        return pa,qa,pa1,qa1

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
        ya,ya1 = BesselFuncs().BSecondOrder(a,x,eps,a,a)
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
            ya = BesselFuncs().ZSERI(a,args[i])
            real = spec.iv(nu,args[i]) ###This looks wrong, it keeps returing nans
           
            if i==0 :
                ax.plot(args[i],ya,label=f'{nu}')
                ax.plot(args[i],real,linestyle='--',label=f'{nu}')
                ax.set_title(labels[i])
                print(f'{labels[i]} % Error: {(((spec.iv(nu,args[i])-ya)/ya).mean())*100} %')

            else:
                print(f'{labels[i]} % Error: {(((spec.iv(nu,args[i])-ya)/ya).mean())*100} %')
    plt.legend()
    plt.show()
    plt.close()
    ######################
    ######ZASYI TEST#####
    #####################
    print('#'*20)
    print(f'{'#'*5} ZASYI Tests {'#'*2}')
    print('#'*20)
    fig, ax  = plt.subplots()
    x = torch.linspace(-100j,0,1000)#[-2,-1,0,1,2]
    I = [0.1j,1j,2j]
    C = [0.1,1+1j,1+2j,2+1j,2+2j]
    args = [x,torch.tensor(I),torch.tensor(C)]
    labels = ['Real','Imaginary','Complex']
  
    for nu in [-.1,0,0.5,1,2]:
        print('#'*10)
        print(f'Order = {nu} \n')
        
        for i in range(3):
            
            a =torch.tensor(nu)
            ya = BesselFuncs().ZASYI(a,args[i]*torch.exp(torch.tensor(1j*torch.pi)),True)#*torch.exp(torch.tensor(1j*torch.pi/2))
            
            if i==0 :
                ax.plot(args[i].imag,ya.abs(),label=f'{nu}')
                ax.plot(args[i].imag,abs(spec.iv(nu,args[i])),linestyle='--',label=f'{nu}')
                ax.set_title(labels[i])
                ax.set_ylim(-10,10)
                print(f'{labels[i]} % Error: {(((ya-spec.iv(nu,args[i]))/ya).mean())*100} %')

            else:
                print(f'{labels[i]} % Error: {(((spec.iv(nu,args[i])-ya)/ya).mean())*100} %')
    plt.legend()
    plt.show()
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
  
    for nu in [0,0.5,1,2]:
        print('#'*10)
        print(f'Order = {nu} \n')
        
        for i in range(3):
            
            a =torch.tensor(nu)
            ya = BesselFuncs().jv(a,args[i])
            
            if i==0 :
                ax.plot(args[i],ya.real,label=f'{nu}')
                ax.plot(args[i],spec.jv(nu,args[i]).real,linestyle='--',label=f'{nu}')
                ax.set_title(labels[i])
                print(f'{labels[i]} % Error: {(((ya-spec.jv(nu,args[i]))/ya).mean())*100} %')

            else:
                print(f'{labels[i]} % Error: {(((spec.jv(nu,args[i])-ya)/ya).mean())*100} %')
    plt.legend()
    plt.show()


tests()


