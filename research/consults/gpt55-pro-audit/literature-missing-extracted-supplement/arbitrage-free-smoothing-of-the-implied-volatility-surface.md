---
source_file: "Arbitrage-free smoothing of the implied volatility surface.pdf"
page_count: 13
extraction_tool: "pypdf 6.10.0 via bundled Codex runtime"
purpose: "Supplement for PDFs present in lit_review but absent from /Volumes/T9/extracted.zip"
---

# Arbitrage-free smoothing of the implied volatility surface


## Page 1

Quantitative Finance
ISSN: 1469-7688 (Print) 1469-7696 (Online) Journal homepage: www.tandfonline.com/journals/rquf20
Arbitrage-free smoothing of the implied volatility
surface
Matthias R. Fengler
To cite this article: Matthias R. Fengler (2009) Arbitrage-free smoothing of the implied volatility
surface, Quantitative Finance, 9:4, 417-428, DOI: 10.1080/14697680802595585
To link to this article:  https://doi.org/10.1080/14697680802595585
Published online: 09 Jun 2009.
Submit your article to this journal
Article views: 2023
View related articles
Citing articles: 19 View citing articles
Full Terms & Conditions of access and use can be found at
https://www.tandfonline.com/action/journalInformation?journalCode=rquf20


## Page 2

Quantitative Finance , Vol. 9, No. 4, June 2009, 417–428
Arbitrage-free smoothing of the implied
volatility surface
MATTHIAS R. FENGLER*
Trading & Derivatives, Sal. Oppenheim jr. & Cie., Untermainanlage 1,
60329 Frankfurt am Main, Germany
(Received 8 July 2005; in final form 17 October 2008 )
The pricing accuracy and pricing performance of local volatility models depends on the
absence of arbitrage in the implied volatility surface. An input implied volatility surface that is
not arbitrage-free can result in negative transition probabilities and consequently mispricings
and false greeks. We propose an approach for smoothing the implied volatility smile in an
arbitrage-free way. The method is simple to implement, computationally cheap and builds on
the well-founded theory of natural smoothing splines under suitable shape constraints.
Keywords: Implied volatility surface; Local volatility; Cubic spline smoothing; No-arbitrage
constraints
1. Introduction
The implied volatility surface (IVS) obtained by inverting
the Black–Scholes (BS) formula serves as a key parameter
for pricing and hedging exotic derivatives. For this
purpose, other models, more sophisticated than the BS
valuation approach, are calibrated to the IVS. A classical
candidate is the local volatility model proposed by
Derman and Kani (1994), Dupire (1994) and Rubinstein
(1994). Local volatility models posit the (risk-neutral)
stock price evolution given by
dSt
St
¼ð rt /C0 /C14 tÞ dt þ /C27 ðSt, tÞdWt, ð1Þ
where Wt denotes a standard Brownian motion, and rt
and /C14 t the continuously compounded interest rate and
a dividend yield, respectively (both assumed to be
deterministic here). Local volatility /C27 ðSt, tÞ is a non-
parametric, deterministic function depending on the asset
price St and time t. A priori unknown, it must be
computed numerically from option prices or, equiva-
lently, from the IVS. Techniques for calibration and
pricing are proposed, among others, by Andersen and
Brotherton-Ratcliffe (1997), Avellaneda et al . (1997),
Dempster and Richards (2000), Jiang and Tao (2001)
and Jiang et al . (2003).
A crucial property of the calibration data, given as an
ensemble of market prices quoted for different strikes and
expiries, is the absence of arbitrage. In this context, we
refer to arbitrage as to any violation of the theoretical
properties of option prices, such as negative butterfly and
calendar spreads (see Section 2 for details). If the market
data admit arbitrage, the calibration of the local volatility
model can fail since negative local volatilities or negative
transition probabilities ensue, which obstructs the con-
vergence of the finite difference schemes solving the
underlying generalized Black–Scholes partial differential
equation. Occasional arbitrage violations may be over-
ridden by an ad hoc approach, but the algorithm fails
when the violations become excessive. While the robust-
ness of the calibration process can be improved by
regularizing techniques (Lagnado and Osher 1997,
Bodurtha and Jermakyan 1999, Cre ´pey 2003a, b), the
specific numerical implementation does not solve the
underlying economic problem of data contaminated with
arbitrage. One may therefore obtain mispricings and
noisy greeks. For illustration, we present in figure 1 the
delta of a down-and-out put which is computed from
a local volatility pricer using a volatility surface contami-
nated by arbitrage (the data are given in Appendix B).
Comparing with a delta calculated from cleaned data
(figure 2) it is apparent that the delta displays local
discontinuities which are – aside from the one at the
barrier – not economically meaningful. The delta position
*Email: matthias.fengler@oppenheim.de
Quantitative Finance
ISSN 1469–7688 print/ISSN 1469–7696 online /C223 2009 Taylor & Francis
http://www.informaworld.com
DOI: 10.1080/14697680802595585


## Page 3

will hence undergo sudden and unforeseeable jumps as the
spot moves, which are inexplicable by changing market
conditions or higher-order greeks. The hedging perfor-
mance can therefore deteriorate dramatically.
Unfortunately, an arbitrage-free IVS is not a natural
situation in practice, since it is often computed from bid
and ask prices, or derived from settlement data of poor
quality, see Hentschel (2003) for an exhaustive exposition
of this topic. As a strategy to overcome this deficiency,
one employs algorithms to remove arbitrage violations
from the raw data. Kahale ´(2004) proposes an interpola-
tion procedure based on piecewise convex polynomials
mimicking the BS pricing formula. The resulting estimate
of the call price function is globally arbitrage-free and so
is the volatility smile computed from it. In a second step,
the total (implied) variance is interpolated linearly along
strikes. Crucially, for the interpolation algorithm to work,
the data must be arbitrage-free from the outset. Instead of
smoothing prices, Benko et al . (2007) suggest to estimate
the IVS using local quadratic polynomials. Their strategy
requires to solve a smoothing problem under nonlinear
constraints.
Here, we propose an approach that unlike Kahale ´
(2004) is based on cubic spline smoothing of option prices
rather than on interpolation. Therefore, the input data
do not have to be arbitrage-free. More specifically, for
a sample of strikes and call prices, fðu
i, yiÞg, ui 2½ a, b/C138for
i ¼ 1, ... , n, we consider the curve estimate defined as
minimizer bg of the penalized sum of squares
Xn
i¼1
wi yi /C0 gðuiÞ
/C8/C9 2
þ/C21
Z b
a
fg00ðvÞg2 dv, ð2Þ
given strictly positive weights w1, ... , wn. The
minimization is carried out with respect to appropriately
chosen, linear constraints. The minimizer bg is a twice
differentiable function and represents a globally arbit-
rage-free call price function, the smoothness of which is
determined by the parameter /C21 4 0. To cope with
calendar arbitrage across different expiries we apply
problem (2) iteratively to each expiry in adding further
constraints. More precisely, we take advantage of
a monotonicity property for European options along
forward-moneyness corrected strike prices. These addi-
tional inequality constraints are straightforward to add to
the minimization. Finally, via the BS formula, one
obtains an IVS well-suited for pricing and hedging.
In employing cubic spline smoothing, we benefit from
a number of nice properties. First, it is possible to cast
problem (2) into a convex quadratic program that is
known to be solvable within polynomial time (Floudas
and Viswewaran 1995). Second, by virtue of convexity, we
have uniqueness of the minimizer. Third, from
a statistical point of view, spline smoothers under shape
constraints achieve optimal rates of convergence in shape-
restricted Sobolev classes (Mammen and Thomas-Agnan
1999). Finally, since the natural cubic spline is entirely
determined by its set of function values and second-order
derivatives at the knots, it can be stored and evaluated at
the desired grid points in an efficient way and interpola-
tion between grid points is unnecessary. In this way, the
method complements existing local volatility pricing
engines. The approach is close to the literature on
estimating risk-neutral transition densities non-parame-
trically, such as Aı ¨t-Sahalia and Duarte (2003) and
Ha¨rdle and Yatchew (2006), but is less complicated and
also applicable when data are scarce (typically there are
20–25 observations, one for each strike, only).
The paper is organized as follows. The next section
outlines the principles of no-arbitrage in the option
pricing function. Section 3 presents spline smoothing
0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
−0.2
−0.1
0
0.1
0.2
0.3
0.4
0.5
Spot moneyness
Figure 1. Delta of a one-year down-and-out put calculated from
arbitrage-contaminated IVS of DAX settlement data from 13
June 2000. Strike is at 120% and barrier at 80% of the DAX
spot price at 7268.91. Pricing follows Andersen and Brotherton-
Ratcliffe (1997) which is an implicit finite difference solver; delta
is read from the grid.
0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
−0.1
0
0.1
0.2
0.3
0.4
0.5
Spot moneyness
Figure 2. Delta of a one-year down-and-out barrier put
calculated from the arbitrage-free IVS of DAX settlement data
from 13 June 2000. Strike is at 120% and barrier at 80% of the
DAX spot at 7268.91. Pricing follows Andersen and Brotherton-
Ratcliffe (1997); delta is read from the grid.
418 M. R. Fengler


## Page 4

under no-arbitrage constraints. In section 4, we explore
some examples and simulations, and section 5 concludes.
2. No-arbitrage constraints on call prices and the IVS
In a dynamically complete market, the absence of
arbitrage opportunities implies the existence of an
equivalent martingale measure (Harrison and Kreps
1979, Harrison and Pliska 1981) that is uniquely
characterized by a risk-neutral transition probability
function. We assume that its density exists, which we
denote by /C30 ðt, T, S
T Þ¼ /C30 ðt, T, ST, frs, /C14 sgt/C20s/C20TÞ, where St is
the time- t asset price, T ¼ t þ /C28 the expiry date of the
option, /C28 time-to-expiration, rt the deterministic risk-free
interest rate and /C14 t a deterministic dividend yield of the
asset. The valuation function of a European call with
strike K is then given by
C
tðK, T Þ¼ e/C0
R T
t
rs ds
Z 1
0
maxðST /C0 K,0 Þ /C30 ðt, T, ST Þ dST:
ð3Þ
From equation (3), the well-known fact that the call
price function is a decreasing and convex function in K is
immediately obtained.y Taking the derivative with respect
to K, and together with the positivity of /C30 and its
integrability to one, one obtains
/C0e/C0
R T
t
rs ds /C20qC
qK /C200, ð4Þ
which implies monotonicity. Convexity follows from
differentiating a second time with respect to K (Breeden
and Litzenberger 1978):
q2C
qK2 ¼ e/C0
R T
t
rs ds/C30 ðt, T, ST Þ/C210: ð5Þ
Finally, by general no-arbitrage considerations, the call
price function is bounded by
max
/C16
e/C0
R T
t
/C14 s dsSt /C0 e/C0
R T
t
rs dsK,0
/C17
/C20CtðK, T Þ/C20e/C0
R T
t
/C14 s dsSt:
ð6Þ
These constraints are clear-cut for the option price
function, but translate into nonlinear conditions for the
implied volatility smile. This can be seen by computing
equation (5) explicitly, using the BS formula and
assuming a strike-dependent implied volatility function,
see Brunner and Hafner (2003) and Benko et al. (2007) for
details.
In the time-to-maturity direction only weak constraints
on the option price function are known. The prices of
American calls for the same strikes must be non-decreas-
ing, which translates to European calls in the absence of
dividends. With non-zero dividends, it can be shown that
there exists a monotonicity property for European call
prices along forward-moneyness corrected strikes.
This result implies that total (implied) variance must be
non-decreasing in forward-moneyness to preclude arbit-
rage. We define total variance as /C23
2ð/C20 , /C28 Þ¼ b/C27 2ð/C20 , /C28 Þ/C28 , where
/C20 ¼ K=FT
t is forward-moneyness and FT
t ¼ Ste
R T
t
ðrs/C0/C14 sÞ ds
the forward price. The BS implied volatility b/C27 is derived
by equating market prices with the BS formula
CBS
t ðK, T Þ¼ e/C0
R T
t /C14 s dsSt/C8 ð /C22d1Þ/C0 e/C0
R T
t rs dsK/C8 ð /C22d2Þ,
where /C8 is the CDF of the standard normal distribution,
and /C22d1 ¼f lnðSt=KÞþ
R T
t ðrs /C0 /C14 sÞ ds þ 1
2 b/C27 2/C28 g=fb/C27 ﬃﬃ ﬃ/C28p g and
/C22d2 ¼ /C22d1 /C0 b/C27 ﬃﬃ ﬃ/C28p . The monotonicity property, which
appears to have been found by a number of practitioners
independently (Gatheral 2004, Kahale ´ 2004, Reiner
2004), must to our knowledge be credited to Reiner
(2000).
Proposition 2.1 (Reiner 2000) : Let r
t be an interest rate
and /C14 t the dividend yield , both depending on time only . For
/C28 1 ¼ T1 /C0 t 5 /C28 2 ¼ T2 /C0 t and two strikes K1 and K2 related
by the forward-moneyness , there is no calendar arbitrage if
CtðK2, T2Þ/C21e
/C0
R T2
T1
/C14 s ds
CtðK1, T1Þ. Furthermore, /C23 2ð/C20 , /C28 iÞ
is an increasing function in /C28 i.
Proof: Given two expiry dates t 5 T1 5 T2, construct
in t the following calendar spread in two calls with the
same forward-moneyness: a long position in the call
CtðK2, T2Þ and a short position in e
/C0
R T2
T1
/C14 s ds
calls CtðK1, T1Þ. The forward-moneyness requirement
implies K1 ¼ e
R T2
T1
ð/C14 s /C0rs Þ ds
K2.I n T1,i f ST1 /C20K1, the calls
in the short position expire out-of-the-money, while
CT1 ðK2, T2Þ/C210. Otherwise, the entire portfolio consists
of CT1 ðK2, T2Þ/C0 e
/C0
R T2
T1
/C14 s ds
(ST1 /C0 e
R T2
T1
ð/C14 s /C0rs Þds
K2)
¼PT1 ðK2,T2Þ/C210 by put-call-parity. Thus, the payoff of
this portfolio is always non-negative. To preclude arbitrage
we must have
CtðK2, T2Þ/C21e
/C0
R T2
T1
/C14 s ds
CtðK1, T1Þ, ð7Þ
which proves the first statement. Multiplying with e
R T2
t
rs ds
and dividing by K2 yields
e
R T2
t
rs dsCtðK2, T2Þ
K2
/C21e
R T1
t
rs dsCtðK1, T1Þ
K1
: ð8Þ
Replacing Ct by CBS
t , define the function
f ð/C20 , /C23 2Þ¼ e
R T
t
rs dsCBS
t ðK, T Þ
K
¼ /C20 /C01/C8 ð /C22d1Þ/C0 /C8 ð /C22d2Þ: ð9Þ
As can be observed, f ð/C20 , /C23 2Þ is a function in /C20 and /C23 2 only
and, for a fixed /C20 , is strictly monotonically increasing in
/C23 2, since @f=@/C23 2 ¼ 1
2 ’ ð /C22d2Þ=
ﬃﬃﬃﬃ ﬃ
/C23 2
p
4 0 for /C23 2 2ð 0, 1Þ. Thus,
yWe stress that these properties do not depend on the existence of a density. In continuous-time models, they hold when the
discounted stock price process is a martingale, but may fail for strict local martingales (Cox and Hobson 2005).
Arbitrage-free smoothing of the implied volatility surface 419


## Page 5

equation (8) implies /C23 2ð/C20 , T2Þ/C21/C23 2ð/C20 , T1Þ, ruling out
calendar arbitrage. œ
To characterize the concept of no-arbitrage in a set of
option data more precisely, we rely on recent work by
Carr et al . (2003), who introduced the concept of ‘static
arbitrage’. Static arbitrage refers to a costless trading
strategy that yields a positive profit with non-zero
probability, but has zero probability to incur a loss.
The term ‘static’ means that positions can only depend on
time and the concurrent underlying stock price.
In particular, they are not allowed to depend on past
prices or on path properties. For a discrete ensemble of
strikes K
i, i ¼ 1, ... , 1 and expiries Tj, j ¼ 1, ... , M,
static no-arbitrage can be established along the line of
arguments outlined in Carr and Madan (2005): given that
the data set does not admit strike arbitrage and calendar
arbitrage, one constructs a convex order of risk-neutral
probability measures at different expiries. The convex
order implies the existence of a Markov martingale by the
results of Kellerer (1972). It follows that there exists
a martingale measure consistent with all call price quotes
and defined on a filtration that contains at least the
underlying asset price and time. Hence the option call
price quotes are free of static arbitrage (Carr et al . 2003,
Carr and Madan 2005).
As a consequence of Proposition 2.1, a plot of the total
variance against the forward moneyness shows calendar
arbitrage when the graphs intersect. In figure 3 we provide
such a total variance plot of our IVS data. Evidently,
there are a significant number of implied volatility
observations with three days to expiry which violate the
no-arbitrage restriction. It is typical that only the front
month violates calendar arbitrage. This occurs when the
short run smile is very pronounced or when the term
structure of the IVS is strongly downward sloping or
humped.
3. Spline smoothing
3.1. Generic set-up
Spline smoothing is a classical statistical technique that is
covered in almost every monograph on smoothing, see for
example Ha¨rdle (1990) and Green and Silverman (1994).
A particularly nice resource is Turlach (2005), whose
exposition we follow closely.
Assume that we observe call prices y
i at strikes
a ¼ u0, ... , unþ1 ¼ b. A function g defined on ½a, b/C138is
called a cubic spline if g, on each subinterval ða, u1Þ,
ðu2, u3Þ, ... , ðun, bÞ, is a cubic polynomial and if g belongs
to the class of twice differentiable functions C2ð½a, b/C138Þ.
The points ui are called knots. The spline g has the
representation
gðuÞ¼
Xn
i¼0
1f½ui, uiþ1Þg siðuÞ, ð10Þ
where siðuÞ¼ ai þ biðu /C0 uiÞþ ciðu /C0 uiÞ2 þ diðu /C0 uiÞ3 for
i ¼ 0, ... , n and given constants ai, bi, ci, di. There are
4ðn þ 1Þ coefficients to be determined. The continuity
conditions on g and its first and second-order derivatives
in each interior segment imply 4 n restrictions on the
coefficients. The indeterminacy can be resolved by
requiring that g has zero second-order derivatives in the
very first and the very last segment of the spline. This
assumption implies that c
0 ¼ d0 ¼ cn ¼ dn ¼ 0, in which
case g is called a natural cubic spline . This choice is
justified by the fact that the gamma of the call converges
fast to zero for high and low strikes. As will be seen
presently, the natural cubic spline allows for a convenient
formulation of the no-arbitrage conditions to be imposed
on the call price function. y
As discussed in Green and Silverman (1994), a more
convenient representation of equation (10) is given by the
so-called value-second derivative representation of the
natural cubic spline. In particular it allows one to
formulate a quadratic program to solve problem (2).
For the value-second derivative representation, put
g
i ¼ gðuiÞ and /C13 i ¼ g00ðuiÞ, for i ¼ 1, ... , n. Furthermore
define g ¼ð g1, ... , gnÞ> and c ¼ð /C13 2, ... , /C13 n/C01Þ>. By defini-
tion, /C13 1 ¼ /C13 n ¼ 0. The non-standard notation of the
0.2 0.4 0.6 0.8 1 1.2 1.4 1.6
0
0.05
0.1
0.15
0.2
0.25
Forward moneyness
Total variance
398
263
198
133
68
48
28
3
Figure 3. Total variance plot for DAX data, 13 June 2000;
see table B1 for the data. Total variance is defined by
/C23
2ðk, /C28 Þ¼ ^/C27 2ð/C20 , /C28 Þ/C28 . Time-to-maturity given in calendar days;
top graph corresponds to top legend entry, second graph to the
second one, etc.
yIt should be noted that the literature on the numerical treatment of splines also discusses other end conditions (Wahba 1990).
A popular choice is to fix the first-order derivatives at the end points of the spline. We experimented with this solution. In this case,
the smoothness penalty no longer has the convenient quadratic form (see Proposition 3.1) but could be approximated by the
smoothness penalty given by the natural spline. Further, since in our application the two first-order derivatives are unknown, they
must be estimated. As proxy we used the first-order BS derivative w.r.t. the strike evaluated at the strike implied volatility. In our
simulations it turned out that the spline functions are very sensitive to a misspecification of the first-order derivatives and less robust
than the natural spline solution.
420 M. R. Fengler


## Page 6

entries in c is proposed by Green and Silverman (1994).
The natural spline is completely specified by the vectors g
and c. In Appendix A, we give the formulae to switch
between the two representations.
Not all possible vectors g and c result in a valid cubic
spline. Sufficient and necessary conditions are formulated
via the following two matrices Q and R. Let h
i ¼ uiþ1 /C0 ui
for i ¼ 1, ... , n /C0 1, and define the n /C2ðn /C0 2Þ matrix Q
by its elements qi, j, for i ¼ 1, ... , n and j ¼ 2, ... , n /C0 1,
given by
qj/C01, j ¼ h/C01
j/C01, qj, j ¼/C0 h/C01
j/C01 /C0 h/C01
j and qjþ1, j ¼ h/C01
j ,
for j ¼ 2, ... , n /C0 1, and qi, j ¼ 0 for ji /C0 jj/C212.
The columns of Q are numbered in the same non-
standard way as the vector c, i.e. the top left element in Q
is q1,2.
The ðn /C0 2Þ/C2ðn /C0 2Þ matrix R is symmetric and is
defined by its elements ri, j for i, j ¼ 2, ... , n /C0 1, given by
ri,i ¼ 1
3 ðhi/C01 þ hiÞ for i ¼ 2, ... , n /C0 1
ri,iþ1 ¼ riþ1,i ¼ 1
6 hi for i ¼ 2, ... , n /C0 2
ri, j ¼ 0 for ji /C0 jj/C212:
ð11Þ
The matrix R is strictly diagonal dominant, so by
standard arguments in linear algebra, R is strictly
positive-definite.
Proposition 3.1: The vectors g and c specify a natural
cubic spline if and only if
Q>g ¼ Rc: ð12Þ
If equation (12) holds, the roughness penalty satisfies
Z b
a
g00ðuÞ2 du ¼ c>Rc: ð13Þ
Proof: See Green and Silverman (1994, section 2.5).
This result allows us to state the spline smoothing task as
a quadratic minimization problem. Define the ð2n /C0 2Þ-
vector y ¼ð w1y1, ... , wnyn,0 , ... ,0 Þ>, where the wi are
strictly positive weights, and the ð2n /C0 2Þ-vector x ¼
ðg>, c>Þ>. Further, define the matrices, A ¼ð Q, /C0 R>Þ and
B ¼ Wn 0
0 /C21 R
/C18/C19
, ð14Þ
where Wn ¼ diagðw1, ... , wnÞ. The solution to problem (2)
can then be written as the solution of the quadratic
program
min
x /C0y>x þ 1
2 x>Bx
subject to A>x ¼ 0:
ð15Þ
The quadratic program (15) serves as the basis for our
arbitrage-free smoothing of the call price function. To this
end we will add further restrictions on x that ensure the
properties outlined in section 2. Since B is strictly positive-
definite by construction, program (15) benefits from two
decisive properties, irrespective of the additional no-
arbitrage constraints to be imposed. First, by positive-
definiteness of B, it belongs to the class of convex
programs which are known to be solvable within
polynomial time (Floudas and Viswewaran 1995).
Algorithms for solving convex quadratic programs are
nowadays available in almost every statistical software
package. An excellent resource is Boyd and
Vandenberghe (2004). Second, and most importantly,
convex programs are known to have a unique minimizer.
Hence the smoothing spline, for given data and /C21 ,i s
unique (Green and Silverman 1994, Theorem 2.4).
3.2. Cubic spline smoothing under no-arbitrage
constraints
It is straightforward to translate the no-arbitrage condi-
tions for the call price function into conditions on the
smoothing spline. Convexity of the spline is imposed by
noting that the second derivative of the spline is linear.
Hence it is sufficient to require that the second derivatives
at the knot points be positive, i.e.
/C13
i /C210, ð16Þ
for i ¼ 2, ... , n /C0 1. By definition, we have /C13 1 ¼ /C13 n ¼ 0.
Next, the price function must be non-increasing in
strikes. Since the convexity constraints insure that the
slope is increasing, it is sufficient to constrain the initial
derivatives at both end points of the spline. For a cubic
spline on the segment ½u
L, uR/C138, the left boundary derivative
from the right is given by g0ðuþ
L Þ¼ð gR /C0 gLÞ=h /C0
hð2/C13 L þ /C13 RÞ=6, and the right boundary derivative from
the left by g0ðu/C0
R Þ¼ð gR /C0 gLÞ=h þ hð/C13 L þ 2/C13 RÞ=6. Thus,
since /C13 1 ¼ /C13 n ¼ 0, the necessary and sufficient constraints
are given by
g2 /C0 g1
h1
/C0 h1
6 /C13 2 /C21/C0e/C0
R T
t
rs ds and
gn /C0 gn/C01
hn/C01
þ hn/C01
6 /C13 n/C01 /C200: ð17Þ
Finally, we add the constraints
e/C0
R T
t
/C14 s dsSt /C0 e/C0
R T
t
rs dsu1 /C20g1 /C20e/C0
R T
t
/C14 s dsSt and gn /C210:
ð18Þ
Including the conditions (16) to (18) into the quadratic
program (15) yields an arbitrage-free call price function
and, in consequence, an arbitrage-free volatility smile.
3.3. Estimating an arbitrage-free IVS
The preceding sections lead to the following natural
procedure to generate an arbitrage-free IVS, which is
outlined in the box.
In order to respect condition (7), Step (i) can be
circumvented by evaluating each spline of the previous
time-to-maturity at the desired strikes. But it might be
faster to employ the initial estimate, because the IVS
observations can easily be spaced on the forward-
moneyness grid. As an initial estimator any two-
dimensional non-parametric smoother, such as a local
polynomial estimator or a thin plate spline, is a natural
Arbitrage-free smoothing of the implied volatility surface 421


## Page 7

candidate (Wahba 1990, Green and Silverman 1994).
The absence of strike arbitrage along the price function
and the absence of calendar arbitrage at the knots is
insured by Step (ii). In general, it cannot be excluded that
there is calendar arbitrage between the knots, but this is
very unlikely given the convex, monotonic, shape of the
call price function.
The smoothing parameter /C21 can either be determined
according to a subjective view, or an automatic, data-
driven, choice of the smoothing parameter can be used.
In the latter case, asymptotically optimal bandwidths
can be found by ‘leave-one-out’ cross-validation techni-
ques (Green and Silverman 1994, section 3.2).
Unfortunately, due to the no-arbitrage constraints
present in the program, the common and efficient
calculation techniques are not applicable. For each
cross validation score it is necessary to solve n separate
smoothing problems, which is cumbersome. However,
the shape constraints we impose – monotonicity and
convexity – act already as a strong smoothing device.
As pointed out by Dole (1999, p. 446), bounds on
second-order derivatives can be seen as smoothing
parameters in their own right. Therefore, the choice of
the smoothing parameter is of secondary importance. It
can be fixed at some small number without large impact
on the estimate (see Turlach 2005 for a related
discussion). Choosing a very small number has the
additional benefit that initially good data will hardly be
smoothed at all.
From the perspective of financial theory, one might
worry that the sum of squared differences of yi /C0 gðuiÞ in
problem (2) may not be the right measure of loss, since an
investor is only interested in relative prices. This concern
can be addressed by using the underlying asset price as
nume´raire. By setting wi ¼ S/C02
t and switching to a spot
moneyness space ~u ¼ u=St, one can conduct the mini-
mization on relative option prices after some obvious
adjustments to the no-arbitrage constraints in quadratic
program (19). The resulting curve estimate ðeg
>,ec>Þ> can
be inflated again via giðuÞ¼ St ~gið ~uÞ and /C13 iðuÞ¼ e/C13 ið ~uÞ=St,
which yields a natural cubic spline as can be verified from
condition (12). Seemingly, this approach comes at the
additional cost of a homogeneity assumption. However,
as can be observed from quadratic program (15), in
choosing as smoothing parameter e/C21 ¼ /C21 S
/C03
t the program
in relative prices is equivalent to the former one in
absolute prices (up to the aforementioned scales). This
property is hidden in the value-second derivative repre-
sentation of the natural cubic spline. For a discussion of
the financial implications of an option pricing function
that is homogeneous in spot and strikes we refer to
Renault (1997), Alexander and Nogueira (2007) and
Fengler et al . (2007).
4. Empirical demonstration
We demonstrate the estimator using single expiries and
the entire IVS of DAX settlement data observed on
13 June 2000; see also table B1 in Appendix B. These data
represent a typical situation one faces when working with
settlement data. By the conditions spelled out in section 2,
market data violating strike arbitrage conditions are
Algorithm
(i) Estimate the IVS via an initial estimate on a regular forward-moneyness grid J¼ ½ /C20 1, /C20 n/C138/C2½t1, tm/C138:
(ii) Iterate through the price surface from the last to the first maturity, and solve the following quadratic
program. For tm, solve
minx /C0y>x þ 1
2 x>Bx,
subject to A>x ¼ 0
/C13 i /C210,
g2 /C0 g1
h1
/C0 h1
6 /C13 2 /C21/C0e
/C0
R T
tm
rs ds
/C0 gn /C0 gn/C01
hn/C01
/C0 hn/C01
6 /C13 n/C01 /C210
g1 /C20e
/C0
R T
tm
/C14 s ds
St ð/C3Þ
g1 /C21e
/C0
R T
tm
/C14 s ds
St /C0 e
/C0
R T
tm
rs ds
u1
gn /C210,
ð19Þ
where x ¼ð g>, c>Þ>;
for tj, j ¼ m /C0 1, ... , 1, solve quadratic program (19) replacing condition ð/C3Þ by
gð j Þ
i 5 e
R tjþ1
tj
/C14 s ds
gð jþ1Þ
i , for i ¼ 1, ... , n,
where gð j Þ
i denotes the ith spline value of maturity j.
422 M. R. Fengler


## Page 8

found by testing in the sample of strikes and prices
ðKi, CiÞ, i ¼ 1, ... , n, whether
/C0e/C0
R T
t
rs ds /C20Ci /C0 Ci/C01
Ki /C0 Ki/C01
/C20Ciþ1 /C0 Ci
Kiþ1 /C0 Ki
/C200 ð20Þ
holds. In the optimization we do not use specific weights
and work in absolute prices. A good initial value x0 for
the quadratic program (19) is given by the observed
market prices; the part in x
0 containing the second-order
derivatives is initialized to 1e-3. The smoothing parameter
is fixed at /C21 ¼1e-7. Implied volatility is computed from
the smoothed call prices.
For the exposition, we pick the expiries with 68 and
398 days time-to-maturity as they have a significant vega.
Figures 4 and 5 show the smoothed implied volatility data
together with the original observations printed as crosses.
We identify the (centre) observations that allow for
arbitrage according to equation (20) by an additional
square. Since the residuals computed as differences
between the raw data and the estimated spline are
sometimes hardly discernible, we further present the
implied volatility residuals in figures 6 and 7 and the
price residuals in figures 8 and 9. Note that all
observations marked with a square are in the positive
0.7 0.8 0.9 1 1.1 1.2 1.3 1.4
18
20
22
24
26
28
30
32
34
36
38
Percent
Spot moneyness
Implied volatility, τ = 68 days
Figure 4. Arbitrage-free implied volatility smile for a time-
to-maturity of 68 days. The estimated function is shown as a curve,
original observations are denoted by crosses. Observations
violating strike arbitrage and belonging to the centre strike
price in equation (20) are identified by additional squares.
0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
20
21
22
23
24
25
26
27
28
29
30
Percent
Spot moneyness
Implied volatility, τ = 398 days
Figure 5. Arbitrage-free implied volatility smile for a time-
to-maturity of 398 days. The estimated function is shown as
a curve, original observations are denoted by crosses.
Observations violating strike arbitrage and belonging to the
centre strike price in equation (20) are identified by additional
squares.
0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
30
20
10
0
10
20
30
40
Basis points
Spot moneyness
Implied volatility residuals, τ = 398 days
Figure 7. Implied volatility residuals for the time-to-maturity of
398 days computed as b/C27 i /C0 bb/C27 i, where bb/C27 i denotes the estimator for
the arbitrage-free implied volatility. Residuals belonging to
observations that previously violated strike arbitrage according
to equation (20) are identified by additional squares.
0.7 0.8 0.9 1 1.1 1.2 1.3 1.4
10
5
0
5
10
15
Implied volatility residuals, τ = 68 days
Basis points
Spot moneyness
Figure 6. Implied volatility residuals for the time-to-maturity of
68 days computed as b/C27 i /C0 bb/C27 i, where bb/C27 i denotes the estimator for
the arbitrage-free implied volatility. Residuals belonging to
observations that previously violated strike arbitrage according
to equation (20) are identified by additional squares.
Arbitrage-free smoothing of the implied volatility surface 423


## Page 9

half plane of the plot. The reason is that the simplest way
to correct three observations for convexity is to pull the
centre observation (marked with a square) downwards
and to correct the observations i /C0 1 and i þ 1 into the
opposite direction. This is the correction the quadratic
program chooses in most cases. The adjustments, which
are necessary to achieve an arbitrage-free set of call prices,
can be substantial. Measured in terms of implied volatility
they amount to around 10 basis points in figure 6 and to
around 30 basis points in figure 7. For the price residuals,
the biggest deviations are observed near-the-money,
where the vega sensitivity is highest.
The entire IVS is given in figure 10. The estimate is
obtained using a thin plate spline as initial estimator on
the forward-moneyness grid J¼ ½ 0:6, 1:25/C138/C2½0:1, 1:6/C138
with 100 grid points altogether and by applying the
arbitrage-free estimation technique from the last to first
time-to-maturity. For these computations the implied
volatility observations with three days to expiry were
deleted from the raw data sample as is regularly suggested
in the literature (Andersen and Brotherton-Ratcliffe 1997,
Bodurtha and Jermakyan 1999, Cre ´pey 2003b). In
figure 2 we present the delta of the down-and-out put
obtained from the local volatility model based on this
arbitrage-free data set. The delta is computed via a finite
difference quotient, directly read from the grid of the
PDE solver. As explained in the introduction, the local
discontinuities vanish when using data smoothed in an
arbitrage-free manner.
To give an idea of the properties of our spline
smoothing approach we do a simulation comparing it
with a benchmark model. As benchmark model we
choose the Heston (1993) model, which is often taken
as the first alternative to local volatility models. Under
a risk-neutral measure, the model is given by
dS
t ¼ð rt /C0 /C14 tÞSt dt þ
ﬃﬃﬃﬃﬃ
Vt
p
St dW1
t ð21Þ
d Vt ¼ /C20 ð/C18 /C0 VtÞ dt þ /C27
ﬃﬃﬃﬃﬃ
Vt
p
dW2
t , ð22Þ
where d W1dW2 ¼ /C26 dt. Unlike spline smoothing, the
Heston model is a parametric model with five parameters
/C20 , /C18 , /C27 , /C26 and the initial variance V0. A comparison
between the two models is essentially a comparison of
the trade-off between variance and bias. Nevertheless, it is
instructive to compare both types of model.
The set-up we used is borrowed from Bliss and
Panigirtzoglou (2002) developed for testing the stability
of state price densities. The idea is to resample from
artificially perturbed data. We consider two cases. First we
fit the Heston model to the observed data using the FFT
pricer by Carr and Madan (1999). From the estimated
parameters we generate implied volatility smiles that are
0.4 0.6 0.8 1 1.2 1.4 0
0.5
1
1.5
2
15
20
25
30
35
40
45
Time–to–maturityForward moneyness
Implied volatility [%]
Figure 10. Estimated arbitrage-free IVS using the constrained
cubic spline applied to an initial estimate coming from a thin
plate spline; DAX settlement data, 13 June 2000.
0.7 0.8 0.9 1 1.1 1.2 1.3 1.4
–1
–0.5
0
0.5
1
1.5
2
Spot moneyness
Call price residuals, τ = 68 days
Figure 8. Call price residuals for the time-to-maturity of 68 days
computed as gi /C0 ^gi, where ^gi denotes the value of the estimated
spline. Residuals belonging to observations that previously
violated strike arbitrage according to equation (20) are identified
by additional squares.
0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
−10
−5
0
5
10
15
Spot moneyness
Call price residuals, τ = 398 days
Figure 9. Call price residuals for the time-to-maturity of
398 days computed as gi /C0 ^gi, where ^gi denotes the value of
the estimated spline. Residuals belonging to observations that
previously violated strike arbitrage according to equation (20)
are identified by additional squares.
424 M. R. Fengler


## Page 10

perturbed by zero mean normal errors. The standard
deviation is chosen between 50 to 10 basis points. Then we
fit both models to the perturbed Heston data. Second, we
use the market data and perturb those. Again both models
are fitted. We look at single expiries only, since the Heston
model displayed too much bias when fitted to the entire
surface. The number of simulations is set to 100. At any
time, the natural cubic spline converged, while the Heston
model occasionally did not; in these cases a new set of
random errors was drawn.
The results are displayed in table B3. The trade-off
between variance and bias is quite obvious in the figures.
In almost every case the RSME (root mean square error)
of the spline model is smaller than the Heston model’s.
Furthermore, when comparing the RMSE* measures,
which present the error w.r.t. the true smile, it is evident
that the Heston model is superior to the spline in terms of
identifying its own model, from which data are generated.
This advantage disappears when the market data are used
and perturbed. Of course, the Heston model cannot
identify the unperturbed market, and the error measures
are of comparable size. This is a significant virtue of the
spline smoother, since as a matter of fact the market
model is unknown; it underpins that the spline smoother
is the natural complement to local volatility pricers, which
aim at best-fitting all market prices.
5. Conclusion
Local volatility pricers require as input an arbitrage-free
implied volatility surface (IVS) – otherwise they can
produce mispricings. This is because arbitrage violations
lead to negative transition probabilities in the underlying
finite difference scheme. In this paper, we propose an
algorithm for estimating the IVS in an arbitrage-free
manner. For a single time-to-maturity the approach
consists in applying a natural cubic spline to the call
price function under suitable linear inequality constraints.
For the entire IVS, we first obtain the fit on a fixed
forward-moneyness grid. Second, the natural spline
smoothing algorithm is applied by stepping from the
last expiry to the first one. This precludes calendar and
strike arbitrage.
The method improves on existing algorithms in three
ways. First, the initial data do not have to be arbitrage-
free from the beginning. Second, the solution is obtained
via a convex quadratic program that has a unique
minimizer. Finally, the estimate can be stored efficiently
via the value-second derivative representation of the
natural spline. Integration into local volatility pricers is
therefore straightforward.
Acknowledgements
The paper represents the author’s personal opinion and
does not reflect the views of Sal. Oppenheim. I thank
Matthias Bode, Tom Christiansen, Enno Mammen,
Christian Menn, Daniel Oeltz, Kay Pilz, Peter
Schwendner, and the anonymous referees for their helpful
suggestions. I am indebted to Eric Reiner for making his
material available to me. Support by the Deutsche
Forschungsgemeinschaft and by the SfB 649 is gratefully
acknowledged.
References
Aı¨t-Sahalia, Y. and Duarte, J., Nonparametric option pricing
under shape restrictions. J. Econom. , 2003, 116, 9–47.
Alexander, C. and Nogueira, L.M., Model-free hedge ratios and
scale-invariant models. J. Banking & Finan. , 2007, 31,
1839–1861.
Andersen, L.B.G. and Brotherton-Ratcliffe, R., The equity
option volatility smile: an implicit finite-difference approach.
J. Comput. Finan. , 1997, 1, 5–37.
Avellaneda, M., Friedman, C., Holmes, R. and Samperi, D.,
Calibrating volatility surfaces via relative entropy minimiza-
tion. Appl. Math. Finan. , 1997, 4, 37–64.
Benko, M., Fengler, M.R. and Ha ¨rdle, W., On extracting
information implied in options. Comput. Statist. , 2007, 22,
543–553.
Bliss, R. and Panigirtzoglou, N., Testing the stability of implied
probability density functions. J. Banking & Finan. , 2002, 26,
381–422.
Bodurtha, J.N. and Jermakyan, M., Nonparametric estimation
of an implied volatility surface. J. Comput. Finan. , 1999, 2,
29–60.
Boyd, S. and Vandenberghe, L., Convex Optimization , 2004
(Cambridge University Press: Cambridge, UK).
Breeden, D. and Litzenberger, R., Price of state-contingent
claims implicit in options prices. J. Bus. , 1978, 51, 621–651.
Brunner, B. and Hafner, R., Arbitrage-free estimation of the
risk-neutral density from the implied volatility smile.
J. Comput. Finan. , 2003, 7, 75–106.
Carr, P., Geman, H., Madan, D.B. and Yor, M., Stochastic
volatility for Le´vy processes. Math. Finan., 2003, 13, 345–382.
Carr, P. and Madan, D.B., Option valuation using the fast
Fourier transform. J. Comput. Finan. , 1999, 2, 61–73.
Carr, P. and Madan, D.B., A note on sufficient conditions for
no arbitrage. Finan. Res. Lett. , 2005, 2, 125–130.
Cox, A.M.G. and Hobson, D.G., Local martingales, bubbles
and option prices. Finan. Stat. , 2005, 9, 477–492.
Cre´pey, S., Calibration of the local volatility in a generalized
Black–Scholes model using Tikhonov regularization. SIAM J.
Math. Anal. , 2003a, 34, 1183–1206.
Cre´pey, S., Calibration of the local volatility in a trinomial tree
using Tikhonov regularization. Inverse Problems , 20003b, 19,
91–127.
Dempster, M.A.H. and Richards, D.G., Pricing American
options fitting the smile. Math. Finan. , 2000, 10, 157–177.
Derman, E. and Kani, I., Riding on a smile. RISK, 1994, 7,
32–39.
Deutsche Bo¨rse, Guide to the Equity Indices of Deutsche Bo ¨rse,
5.12 ed., 2006 (Deutsche Bo ¨rse AG: 60485 Frankfurt am
Main: Germany).
Dole, D., CoSmo: a constrained scatterplot smoother for
estimating convex, monotonic transformations. J. Bus. &
Econom. Statist. , 1999, 17, 444–455.
Dupire, B., Pricing with a smile. RISK, 1994, 7, 18–20.
Fengler, M.R., Ha¨rdle, W. and Mammen, E., A semiparametric
factor model for implied volatility surface dynamics. J. Finan.
Econom., 2007, 5, 189–218.
Floudas, C.A. and Viswewaran, V., Quadratic optimization,
In Handbook of Global Optimization , edited by R. Horst and
P.M. Pardalos, pp. 217–270, 1995 (Kluwer Academic:
Dordrecht, The Netherlands).
Arbitrage-free smoothing of the implied volatility surface 425


## Page 11

Gatheral, J., A parsimonious arbitrage-free implied volatility
parametrization with application to the valuation of volatility
derivatives [online], 2004. Available online at: www.math.
nyu.edu/fellows_fin_math/gatheral/madrid2004.pdf
Green, P.J. and Silverman, B.W., Nonparametric Regression
and Generalized Linear ModelsMonographs on Statistics and
Applied Probability , Vol. 58, 1994 (Chapman & Hall:
London).
Ha¨rdle, W., Applied Nonparametric Regression , 1990
(Cambridge University Press: Cambridge, UK).
Ha¨rdle, W. and Yatchew, A., Dynamic state price density
estimation using constrained least squares and the bootstrap.
J. Econom. , 2006, 133, 579–599.
Harrison, J. and Kreps, D., Martingales and arbitrage in multi-
period securities markets. J. Econ. Theory , 1979, 20, 381–408.
Harrison, J. and Pliska, S., Martingales and stochastic integrals
in the theory of continuous trading. Stochast. Process. &
Appls, 1981, 11, 215–260.
Hentschel, L., Errors in implied volatility estimation. J. Finan. &
Quant. Anal. , 2003, 38, 779–810.
Heston, S., A closed-form solution for options with stochastic
volatility with applications to bond and currency options.
Rev. Finan. Studies , 1993, 6, 327–343.
Jiang, L., Chen, Q., Wang, L. and Zhang, J.E., A new well-
posed algorithm to recover implied local volatility. Quant.
Finan., 2003, 3, 451–457.
Jiang, L. and Tao, Y., Identifying the volatility of the underlying
assets from option prices. Inverse Problems , 2001, 17,
137–155.
Kahale´, N., An arbitrage-free interpolation of volatilities. RISK,
2004, 17, 102–106.
Kellerer, H.G., Markov-Komposition und eine Anwendung auf
Martingale. Math. Annalen , 1972, 198, 99–122.
Lagnado, R. and Osher, S., A technique for calibrating
derivative security pricing models: numerical solution of an
inverse problem. J. Comput. Finan. , 1997, 1, 13–25.
Mammen, E. and Thomas-Agnan, C., Smoothing splines and
shape restrictions. Scand. J. Statist. , 1999, 26, 239–252.
Reiner, E., Calendar spreads, characteristic functions, and
variance interpolation. Mimeo, 2000.
Reiner, E., The characteristic curve approach to arbitrage-free
time interpolation of volatility. Presentation at the ICBI
Global Derivatives and Risk Management Conference,
Madrid, Espan˜a, 2004.
Renault, E., Econometric models of option pricing errors,
In Advances in Economics and Econometrics, Seventh World
Congress, edited by D.M. Kreps and K.F. Wallis,
Econometric Society Monographs, pp. 223–278, 1997
(Cambridge University Press: Cambridge, UK).
Rubinstein, M., Implied binomial trees. J. Finan. , 1994, 49,
771–818.
Turlach, B.A., Shape constrained smoothing using smoothing
splines. Comput. Statist. , 2005, 20, 81–104.
Wahba, G., Spline Models for Observational Data , 1990 (SIAM:
Philadelphia, PA).
Appendix A: Transformation formulae
To switch from the value-second derivative representation
to the piecewise polynomial representation (10), employ
ai ¼ gi
bi ¼ giþ1 /C0 gi
hi
/C0 hi
6 ð2/C13 i þ /C13 iþ1Þ
ci ¼ /C13 i
2
di ¼ /C13 iþ1 /C0 /C13 i
6hi
ðA1Þ
for i ¼ 1, ... , n /C0 1. Furthermore,
a0 ¼ a1 ¼ g1, an ¼ gn, b0 ¼ b1, c0 ¼ d0 ¼ cn ¼ dn ¼ 0,
and
bn ¼ s0
n/C01ðunÞ¼ bn/C01 þ 2cn/C01hn/C01 þ 3dn/C01h2
n/C01
¼ gn /C0 gn/C01
hn/C01
þ hi
6 ð/C13 n/C01 þ 2/C13 nÞ,
where hi ¼ uiþ1 /C0 ui for i ¼ 1, ... , n /C0 1 and /C13 1 ¼ /C13 n ¼ 0.
Changing vice versa is accomplished by
gi ¼ siðuiÞ¼ ai for i ¼ 1, ... , n,
/C13 i ¼ s00
i ðuiÞ¼ 2ci for i ¼ 2, ... , n /C0 1,
/C13 1 ¼ /C13 n ¼ 0:
ðA2Þ
Appendix B: Data
Table B1. Raw DAX implied volatility data from 13 June 2000, traded at the EUREX, Germany. Time-to-maturity measured in
calendar days.
Implied volatilities
Time-to-maturity
strikes 3 28 48 68 133 198 263 398
2600 367.09
2800 340.92
3000 316.57
3200 293.81
3400 272.43
3600 252.28
3800 233.23
4000 215.16
4200 197.98 38.39
4400 181.60 37.10
4600 165.96 36.19 34.51
4800 150.99 36.86 35.25 33.79
4900 143.73 36.15
5000 136.63 35.65 34.16 33.28
5100 129.66 35.38
5200 122.83 34.57 33.55 32.32
(continued)
426 M. R. Fengler


## Page 12

Table B1. Continued.
Implied volatilities
Time-to-maturity
strikes 3 28 48 68 133 198 263 398
5300 116.13 33.94
5400 109.56 33.57 32.35 31.82
5500 103.11 33.02
5600 96.78 31.86 32.30 31.70 31.11
5700 90.56 31.80
5800 84.45 30.18 31.53 30.65 30.16
5900 78.44 30.73
6000 72.54 28.46 29.12 30.13 29.92 29.38 29.47 29.49
6100 66.74 29.77
6200 61.03 27.04 28.01 29.17 29.04 28.72 28.91 28.78
6250 58.21
6300 55.42 28.54 28.58
6350 52.64
6400 55.24 25.86 27.06 28.19 28.26 28.06 28.00 27.83
6450 52.88
6500 50.76 27.66 28.04
6550 48.08
6600 45.41 24.70 26.02 27.08 27.44 27.24 27.39 27.87
6650 42.77 24.29 25.74 26.88
6700 40.15 24.15 25.38 26.74 26.99 26.97
6750 37.55 24.05 25.12 26.59
6800 34.97 23.59 24.96 26.23 26.67 26.79 26.79 27.15
6850 34.14 23.32 24.90 25.92
6900 31.62 23.21 24.49 25.69 26.48 26.30
6950 29.71 23.00 24.21 25.54
7000 28.95 22.62 24.02 25.45 25.91 25.88 26.00 26.63
7050 28.46 22.41 23.94 25.15
7100 26.85 22.42 23.65 24.83 25.50 25.53
7150 27.11 22.01 23.34 24.59
7200 25.56 21.74 23.14 24.43 25.24 25.30 25.54 26.32
7250 25.30 21.69 23.07 24.34
7300 23.98 21.21 22.65 23.91 24.87 24.86
7350 23.80 20.94 22.33 23.60
7400 23.59 20.86 22.16 23.37 24.39 24.40 24.47 24.94
7450 23.91 20.77 22.11 23.23
7500 24.87 20.46 21.88 23.17 24.06 24.06
7550 25.59 20.37 21.62 22.95
7600 26.96 20.37 21.48 22.67 23.89 23.85 23.90 24.45
7650 28.32 20.06 21.48 22.48
7700 29.03 20.00 22.38 23.51 23.71
7750 30.25 20.10 22.35
7800 32.47 19.80 20.93 22.07 23.14 23.30 23.56 24.15
7850 34.69 21.84
7900 36.37 19.89 20.65 21.72 22.93 22.97
7950 35.67 21.67
8000 37.85 19.55 20.48 21.48 22.76 22.76 22.91 23.67
8050 40.02 21.32
8100 42.17 21.15 22.35 22.66
8150 44.31
8200 46.44 19.53 20.04 21.08 22.13 22.33 22.59 23.19
8250 48.55
8300 20.68 22.06 22.02
8400 54.81 19.54 20.60 21.69 21.81 22.05 22.92
8500 21.72
8600 62.97 19.38 20.23 21.36 21.45
8700 21.14
8800 70.95 19.93 20.85 20.97 21.31 22.07
8900 20.90
9000 78.75 19.78 20.61 20.63
9100 20.37
9200 86.37 19.74 20.25 20.21 20.64 21.46
9400 93.84 19.91 19.93 19.86
9600 101.14 19.93 19.86 19.52 19.90 20.85
9800 108.29 19.94 19.54 19.15
10000 115.30 20.38 19.53 18.89 19.19 20.13
10200 19.35
10400 19.41
Arbitrage-free smoothing of the implied volatility surface 427


## Page 13

Table B3. RMSE (root mean square error) for prices and implied volatilities for the natural cubic spline (NCS) and the Heston
model computed from unweighted observations. Number of simulations is 100. RMSE* is the error between the true (or the market)
model and the perturbed one. Last line gives the standard deviation of the errors added to implied volatility during the simulations.
Time-to-maturity 3 28 48 68 133 198 263 398
NCS, RMSE (prices) 1.0749 0.3542 0.3552 0.3648 0.5623 0.6617 1.4107 3.9284
NCS, RMSE (vol) 0.0496 0.0004 0.0003 0.0004 0.0004 0.0003 0.0006 0.0012
Heston, RMSE (prices) 4.1435 0.7916 0.9091 1.0389 1.7211 2.5468 3.8228 8.1583
Heston, RMSE (vol) 0.4684 0.0040 0.0008 0.0031 0.0034 0.0017 0.0019 0.0026
/C20 1.0066 9.0638 7.6691 1.2101 0.4868 1.0146 0.1302 0.1254
/C18 9.2017 0.0682 0.0636 0.4323 0.5882 0.2278 1.1336 0.8087
/C27 4.3041 0.8755 0.8902 0.8300 0.6445 0.6286 0.5432 0.4503
/C26 /C00.3019 /C00.4399 /C00.5243 /C00.6084 /C00.6592 /C00.6892 /C00.7037 /C00.7327
V
0 0.0001 0.0370 0.0492 0.0011 0.0010 0.0001 0.0003 0.0014
Simulation from estimated Heston
NCS, RMSE (prices) 0.0275 0.5873 1.1637 0.6514 1.5852 2.3430 0.8430 1.4461
NCS, RMSE (vol) 0.0039 0.0031 0.0025 0.0009 0.0013 0.0014 0.0004 0.0005
NCS, RMSE* (prices) 0.5311 1.6894 2.1193 1.5639 2.2984 2.8380 1.5578 2.0008
NCS, RMSE* (vol) 0.0055 0.0045 0.0039 0.0020 0.0019 0.0017 0.0008 0.0007
Heston, RMSE (prices) 0.4668 1.6949 2.4265 1.7513 2.9378 3.9392 1.8838 2.6694
Heston, RMSE (vol) 0.0067 0.0079 0.0066 0.0026 0.0025 0.0023 0.0009 0.0010
Heston, RMSE* (prices) 0.3348 0.9430 1.1195 0.6452 0.8520 0.8915 0.4786 0.5445
Heston, RMSE* (vol) 0.0044 0.0058 0.0046 0.0014 0.0009 0.0006 0.0003 0.0002
Simulation from observed market data
NCS, RMSE (prices) 1.3195 2.5556 3.7661 1.9850 2.4999 3.7458 1.8047 4.3230
NCS, RMSE (vol) 0.0395 0.0034 0.0034 0.0018 0.0016 0.0017 0.0008 0.0014
NCS, RMSE* (prices) 1.3510 2.0523 2.7571 1.5376 2.3321 3.2912 2.2842 4.6488
NCS, RMSE* (vol) 0.0393 0.0035 0.0028 0.0017 0.0018 0.0017 0.0010 0.0015
Heston, RMSE (prices) 4.1974 3.6398 4.7761 3.4267 5.0641 5.6090 4.4932 8.8147
Heston, RMSE (vol) 0.4794 0.0107 0.0046 0.0090 0.0081 0.0029 0.0020 0.0028
Heston, RMSE* (prices) 4.1438 1.7627 1.6052 2.2388 3.7631 2.9330 3.9613 8.7661
Heston, RMSE* (vol) 0.4793 0.0093 0.0017 0.0084 0.0076 0.0019 0.0018 0.0028
St. Dev. of simul. errors (basis points) 50 50 50 25 25 25 10 10
Table B2. Data of zero rates are obtained from EURIBOR quotes from 13 June 2000. Time-to-maturity is given in calendar days.
For pricing, the dividend yield is assumed to be zero, since the DAX index is a performance index, see Deutsche Bo ¨rse (2006) for
a precise description. DAX spot price is St ¼ 7268.91.
Time-to-maturity 3 28 48 68 133 198 263 398
Interest rate 4.36% 4.47% 4.53% 4.57% 4.71% 4.85% 4.93% 5.04%
428 M. R. Fengler
