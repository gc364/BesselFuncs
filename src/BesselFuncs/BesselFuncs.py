import torch
import math

def Pochhammer(nu,k):
    """Also called a rising factorial power"""
    if k==0:
        return 1
    num=1
    s = 0
    for i in range(1,2*k,2):
        
        num*=((4*nu**2)-(i**2))
        s+=1/((4*nu**2)-(i**2))
    den = 2**(3*k)*torch.special.gammaln(torch.tensor(k+1)).exp()

    return num/den
    


def ZASYI(nu,z,BR=False):
    """Modified Bessel function 1st kind using the asymptotic expansion for large abs(z)
    Only valid for positve abs(z)
    Uses DMLF 10.4.E.5"""
    N = math.ceil(max(abs(nu)-0.5,1))
    
    norm2 = torch.exp(z)/torch.sqrt(2*torch.pi*z)
    norm1 = (1j*torch.exp(-z+torch.pi*nu*1j))/torch.sqrt(2*torch.pi*z)
    if BR:
        norm1 = (1j*torch.exp(-z-torch.pi*nu*1j))/torch.sqrt(2*torch.pi*z)
    t1=0
    t2=0
    for k in range(N):
        a = Pochhammer(nu,k)
        
        #Term 1
        t1+=a/((z)**k)
        #Term 2
        t2+= ((-1)**(k)*a)/((z)**k)

    return norm1*t1+norm2*t2
    

def ZSERI(nu,z):
    """Power series expansion for calculating modified bessel functions 1st kind with complex argument
    Valid for abs(z) = [-2,2]"""
    
    s = 0
    for k in range(10):
        num = (z/2)**(2*k+nu)
        denom = torch.special.gammaln(torch.tensor(k+1)).exp()*torch.special.gammaln(nu+k+1).exp()
        s+=num/denom
    return s


def jv(nu,x)->torch.Tensor:
    """Reimplementation of scipy jv using pytorch.
    Follows AMOS FORTRAN implementation zbesj.f
    Bessel function of the first kind"""
    
    if len(nu.shape)>1:
        raise ValueError('nu must be a 1D tensor')
    elif len(nu.shape)==0:
        return _jv(nu,x)

    J = torch.zeros_like(x,dtype=torch.complex128)

   
    for n in range(nu.shape[0]):
        xp = x[:,n]
        nup = nu[n]
        Jp = _jv(nup,xp)
        J[:,n] = Jp
    
    return J


def _jv(nu,x,eps=1e-12,ya=0,ya1=0):
    ##This takes a scalar nu and 1D tensor x
    
    J = torch.zeros_like(x,dtype=torch.complex128)
    x = x.to(torch.complex128)
    if torch.sign(nu) != -1.:
        for zi in range(x.shape[0]):
            
            if x[zi].real <=0: #If the real part of z is -ve then the condition on the imaginary part has to change. No clue why
                if x[zi].imag >= 0:  
                    I= ZBINU(nu,-1j*x[zi])
                    J[zi] = torch.exp((nu*torch.pi*1j)/2)*I
                else:
                    I= ZBINU(nu,1j*x[zi])
                    J[zi] = torch.exp((-nu*torch.pi*1j)/2)*I
            else:
                if x[zi].imag > 0:   
                    I= ZBINU(nu,-1j*x[zi])
                    J[zi] = torch.exp((nu*torch.pi*1j)/2)*I
                else:
                    I= ZBINU(nu,1j*x[zi])
                    J[zi] = torch.exp((-nu*torch.pi*1j)/2)*I

        return J
    else:
        ##This reflects the order with a Bessel function of the second kind
        nu = -nu
        for zi in range(x.shape[0]):
        
            if x[zi].real <=0: #If the real part of z is -ve then the condition on the imaginary part has to change. No clue why
                if x[zi].imag >= 0:  
                    I= ZBINU(nu,-1j*x[zi])
                    J[zi] = torch.exp((nu*torch.pi*1j)/2)*I
                else:
                    I= ZBINU(nu,1j*x[zi])
                    J[zi] = torch.exp((-nu*torch.pi*1j)/2)*I
            else:
                if x[zi].imag > 0:   
                    I= ZBINU(nu,-1j*x[zi])
                    J[zi] = torch.exp((nu*torch.pi*1j)/2)*I
                else:
                    I= ZBINU(nu,1j*x[zi])
                    J[zi] = torch.exp((-nu*torch.pi*1j)/2)*I
            J[zi] = J[zi]*torch.cos(torch.pi*nu)-ZBESY(nu,x[zi])*torch.sin(torch.pi*nu)
        return J


    
def ZBINU(nu,z):
    """Conglomeration of all the methods of computing modified bessel functions of the 1st kind.
    Will choose the appropiate one given the values of z"""
    
    if (torch.sign(torch.min(z.real)) == -1 ) :
        if (torch.abs(z)<=5) :
            return ZSERI(nu,z)
        else:
            #Calculate in the right half of the complex plane
            rCp =  ZASYI(nu,torch.exp(torch.tensor(1j*torch.pi))*z)
            return rCp *torch.exp(nu*torch.pi)##spin around to the left
    else:
        if (torch.abs(z)<=5) :
            return ZSERI(nu,z)
        else:
            return ZASYI(nu,z)
    
def ZBESY(nu,z):
    """Computes the Bessel function of the second kind for complex argument and Real order.
    Requires Hankel functions where M=1 and M=2
        
        Y(FNU,Z)=0.5*(H(1,FNU,Z)-H(2,FNU,Z))/I """
    norm = 0.5/1j
    return norm*(ZBESH(1,nu,z)-ZBESH(2,nu,z))
def ZBESH(m,nu,z):
    """ Computes the Hankel functions for complex argument and Real order."""
    
    if nu>=0:
        if m ==1:
            norm = 2/(torch.pi*1j)
            t1 = torch.exp(-1j*torch.pi*nu/2)*kv(nu,z*torch.exp(torch.tensor(-torch.pi*1j/2)))
        if m==2:
            norm=-2/(torch.pi*1j)
            t1 =  torch.exp(1j*torch.pi*nu/2)*kv(nu,z*torch.exp(torch.tensor(torch.pi*1j/2)))
        H= norm*t1
    else:
        nu=-nu
        if m ==1:
            norm = 2/(torch.pi*1j)
            t1 = torch.exp(-1j*torch.pi*nu/2)*kv(nu,z*torch.exp(torch.tensor(-torch.pi*1j/2)))
            H= norm*t1
            H = H*torch.exp(torch.pi*1j*nu)
        if m==2:
            norm=-2/(torch.pi*1j)
            t1 =  torch.exp(1j*torch.pi*nu/2)*kv(nu,z*torch.exp(torch.tensor(torch.pi*1j/2)))
            H= norm*t1
            H = H*torch.exp(-torch.pi*1j*nu)
        
    
    
    return H

def kv(nu,z):
    """Computes the modified Bessel function of the second kind for complex argument and real order"""
    #TODO: This needs work, this relation holds as abs(z)->\infty, but breaks down near zero. Need to implement the
    # expnasions used by AMOS
    # if torch.abs(z)>5 or torch.abs(z)<1:
    #     return torch.sqrt(torch.pi/(2*z))*torch.exp(-z)
    # else: 
    #     return ZASYK(nu,z)
    return torch.sqrt(torch.pi/(2*z))*torch.exp(-z)

def ZASYK(nu,z):
    """Asymptotic expansion for K-Bessel function, DMLF 10.4.E.2"""
    N = math.ceil(max(abs(nu)-0.5,1))+10
    t=0
    norm = torch.sqrt(torch.pi/2*z)*torch.exp(-z)
    for k in range(N):
        a = Pochhammer(nu,k)
        t+=a/((z)**k)
    return norm*t
    

def jvp(v,z,n)->torch.Tensor:
    if len(v.shape)>1:
        raise ValueError('v must be a 1D tensor')
    elif len(v.shape)==0:
        return _bessel_diff_formula(v,z,n,_jv,-1)

    Jvp = torch.zeros_like(z,dtype=torch.complex128)

    for k in range(v.shape[0]):
        xp = z[:,k]
        nup = v[k]
        Jp = _bessel_diff_formula(nup,xp,n,_jv,-1)
        Jvp[:,k] = Jp
    
    return Jvp

def _bessel_diff_formula(v, z, n, L, phase):
    """Taken directly from SciPy"""
    
    # from AMS55.
    # L(v, z) = J(v, z), Y(v, z), H1(v, z), H2(v, z), phase = -1
    # L(v, z) = I(v, z) or exp(v*pi*i)K(v, z), phase = 1
    # For K, you can pull out the exp((v-k)*pi*i) into the caller
    
    p = 1.0
    s = L(v-n, z)
    for i in range(1, n+1):
        p = phase * (p * (n-i+1)) / i   # = choose(k, i)
        s += p*L(v-n + i*2, z)
    return s / (2.**n)

def BSecondOrder(a,xf,eps,ya,ya1):
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
            s = 1 if abs(e)<1e-15 else sinh(e)/e
            e = torch.exp(e)
            
            recip,p,q = reiprocal_gamma(a,p,q)
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
            p,q,b,h = besspqa(a,x,eps,p,q,b,h)
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
def sinh(x):
    ax = abs(x)
    if ax<.3:
        y = x**2 if ax<.1 else x*x/9
        x = (((1/5040*y +1/120)*y+1/6)*y+1)*x
        sinh = x if ax<.1 else x*(1+4*x*x/27)
    else:
        ax = torch.exp(ax)
        sinh = torch.sign(x)*.5*(ax-1/ax)
    return sinh
def reiprocal_gamma(x,odd,even):
    
    
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

def besspqa(a,x,eps,pa,qa,pa1,qa1):
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




