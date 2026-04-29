---
source_file: "Econometrica - 2011 - Hansen - The Model Confidence Set.pdf"
page_count: 45
extraction_tool: "pypdf 6.10.0 via bundled Codex runtime"
purpose: "Supplement for PDFs present in lit_review but absent from /Volumes/T9/extracted.zip"
---

# Econometrica - 2011 - Hansen - The Model Confidence Set


## Page 1

Econometrica, Vol. 79, No. 2 (March, 2011), 453–497
THE MODEL CONFIDENCE SET
BY PETER R. HANSEN,A SGER LUNDE, AND JAMES M. NASON1
This paper introduces the model conﬁdence set (MCS) and applies it to the selection
of models. A MCS is a set of models that is constructed such that it will contain the
best model with a given level of conﬁdence. The MCS is in this sense analogous to
a conﬁdence interval for a parameter. The MCS acknowledges the limitations of the
data, such that uninformative data yield a MCS with many models, whereas informative
data yield a MCS with only a few models. The MCS procedure does not assume that a
particular model is the true model; in fact, the MCS procedure can be used to compare
more general objects, beyond the comparison of models. We apply the MCS procedure
to two empirical problems. First, we revisit the inﬂation forecasting problem posed by
Stock and Watson (1999), and compute the MCS for their set of inﬂation forecasts.
Second, we compare a number of T aylor rule regressions and determine the MCS of
the best regression in terms of in-sample likelihood criteria.
K
EYWORDS: Model conﬁdence set, model selection, forecasting, multiple compar-
isons.
1. INTRODUCTION
ECONOMETRICIANS OFTEN FACE a situation where several models or meth-
ods are available for a particular empirical problem. A relevant question is,
“Which is the best?” This question is onerous for most data to answer, espe-
cially when the set of competing alternatives is large. Many applications will
not yield a single model that signiﬁcantly dominates all competitors because
the data are not sufﬁciently informative to give an unequivocal answer to this
question. Nonetheless, it is possible to reduce the set of models to a smaller set
of models—a model conﬁdence set—that contains the best model with a given
level of conﬁdence.
The objective of the model conﬁdence set (MCS) procedure is to determine
the set of models,
M∗/commaorithat consists of the best model(s) from a collection of
models, M0/commaoriwhere best is deﬁned in terms of a criterion that is user-speciﬁed.
The MCS procedure yields a model conﬁdence set, ˆM∗, that is a collection of
models built to contain the best models with a given level of conﬁdence. The
process of winnowing models out of M0 relies on sample information about
1The authors thank Joe Romano, Barbara Rossi, Jim Stock, Michael Wolf, and seminar par-
ticipants at several institutions and the NBER Summer Institute for valuable comments, and
Thomas T rimbur for sharing his code for the Baxter–King ﬁlter. The Ox language of Doornik
(2006) was used to perform the calculations reported here. The ﬁrst two authors are grateful for
ﬁnancial support from the Danish Research Agency, Grant 24-00-0363, and thank the Federal
Reserve Bank of Atlanta for its support and hospitality during several visits. The views in this
paper should not be attributed to either the Federal Reserve Bank of Philadelphia or the Federal
Reserve System, or any of its staff. The Center for Research in Econometric Analysis of Time
Series (CREATES) is a research center at Aarhus University funded by the Danish National
Research Foundation.
© 2011 The Econometric Society DOI: 10.3982/ECTA5771


## Page 2

454 P . R. HANSEN, A. LUNDE, AND J. M. NASON
the relative performances of the models inM0. This sample information drives
the MCS to create a random data-dependent set of models, ˆM∗. The set ˆM∗
includes the best model(s) with a certain probability in the same sense that a
conﬁdence interval covers a population parameter.
An attractive feature of the MCS approach is that it acknowledges the lim-
itations of the data. Informative data will result in a MCS that contains only
the best model. Less informative data make it difﬁcult to distinguish between
models and may result in a MCS that contains several (or possibly all) mod-
els. Thus, the MCS differs from extant model selection criteria that choose a
single model without regard to the information content of the data. Another
advantage is that the MCS procedure makes it possible to make statements
about signiﬁcance that are valid in the traditional sense—a property that is not
satisﬁed by the commonly used approach of reporting p-values from multiple
pairwise comparisons. Another attractive feature of the MCS procedure is that
it allows for the possibility that more than one model can be the best, in which
case
M∗ contains more than a single model.
The contributions of this paper can be summarized as follows: First, we in-
troduce a model conﬁdence set procedure and establish its theoretical prop-
erties. Second, we propose a practical bootstrap implementation of the MCS
procedure for a set of problems that includes comparisons of forecasting mod-
els evaluated out of sample and regression models evaluated in sample. This
implementation is particularly useful when the number of objects to be com-
pared is large. Third, the ﬁnite sample properties of the bootstrap MCS proce-
dure are analyzed in simulation studies. Fourth, we apply the MCS procedure
to two empirical applications. We revisit the out-of-sample prediction problem
of Stock and Watson (1999) and construct MCSs for their inﬂation forecasts.
We also build a MCS for T aylor rule regressions using three likelihood criteria
that include the Akaike information criterion (AIC) and Bayesian information
criterion (BIC).
1.1. Theory of Model Conﬁdence Sets
We do not treatmodels as sacred objects; neither do we assume that a partic-
ular model represents the true data generating process. Models are evaluated
in terms of a user-speciﬁed criterion function. Consequently, the “best” model
is unlikely to be replicated for all criteria. Also, we use the term “models”
loosely. It can refer to econometric models, competing forecasts, or alterna-
tives that need not involve any modelling of data, such as trading rules. So the
MCS procedure is not speciﬁc to comparisons of models. For example, one
could construct a MCS for a set of different “treatments” by comparing sam-
ple estimates of the corresponding treatment effects or construct a MCS for
trading rules with the best Sharpe ratio.
A MCS is constructed from a collection of competing objects,
M0,a n da
criterion for evaluating these objects empirically. The MCS procedure is based
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 3

THE MODEL CONFIDENCE SET 455
on an equivalence test , δM,a n da n elimination rule, eM/periodoriThe equivalence test
is applied to the set M = M0.I f δM is rejected, there is evidence that the
objects in M are not equally “good” and eM is used to eliminate from M
an object with poor sample performance. This procedure is repeated until δM
is “accepted” and the MCS is now deﬁned by the set of “surviving” objects.
By using the same signiﬁcance level, α, in all tests, the procedure guarantees
that limn→∞ P(M∗ ⊂ ˆM∗
1−α) ≥ 1 − α; in the case where M∗ consists of one
object, we have the stronger result that limn→∞ P(M∗ = ˆM∗
1−α) = 1/periodoriThe MCS
procedure also yields p-values for each of the objects. For a given object, i ∈
M0, the MCS p-value, ˆpi, is the threshold at which i ∈ ˆM∗
1−α if and only if
ˆpi ≥ α. Thus, an object with a small MCS p-value makes it unlikely that it is
one of the best alternatives in M0/periodori
The idea behind the sequential testing procedure that we use to construct the
MCS may be recognized by readers who are familiar with the trace-test proce-
dure for selecting the rank of a matrix. This procedure involves a sequence of
trace tests (see Anderson (1984)), and is commonly used to select the number
of cointegration relations within a vector autoregressive model (see Johansen
(1988)). The MCS procedure determines the number of superior models in
the same way the trace test is used to select the number of cointegration rela-
tions. A key difference is that the trace-test procedure has a natural ordering
in which the hypotheses are to be tested, whereas the MCS procedure requires
a carefully chosen elimination rule to deﬁne the sequence of tests. We discuss
this issue and related testing procedures in Section 4.
1.2. Bootstrap Implementation and Simulation Results
We propose a bootstrap implementation of the MCS procedure that is con-
venient when the number of models is large. The bootstrap implementation is
simple to use in practice and avoids the need to estimate a high-dimensional
covariance matrix. White (2000b) is the source of many of the ideas that un-
derlies our bootstrap implementation.
We study the properties of our bootstrap implementation of the MCS pro-
cedure through simulation experiments. The results are very encouraging be-
cause the best model does end up in the MCS at the appropriate frequency and
the MCS procedure does have power to weed out all the poor models when the
data contain sufﬁcient information.
1.3. Empirical Analysis of Inﬂation Forecasts and T aylor Rules
We apply the MCS to two empirical problems. First, the MCS is used to
study the inﬂation forecasting problem. The choice of an inﬂation forecast-
ing model is an especially important issue for central banks, treasuries, and
private sector agents. The 50-plus year tradition of the Phillips curve suggests
it remains an effective vehicle for the task of inﬂation forecasting. Stock and
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 4

456 P . R. HANSEN, A. LUNDE, AND J. M. NASON
Watson (1999) made the case that “a reasonably speciﬁed Phillips curve is the
best tool for forecasting inﬂation”; also seeGordon (1997), Staiger, Stock, and
Watson (1997b), and Stock and Watson (2003). Atkeson and Ohanian (2001)
concluded that this is not the case because they found it is difﬁcult for any of
the Phillips curves they studied to beat a simple no-change forecast in out-of-
sample point prediction.
Our ﬁrst empirical application is based on the Stock and Watson (1999)
data set. Several interesting results come out of our analysis. We partition
the evaluation period in the same two subsamples as did Stock and Watson
(1999). The earlier subsample covers a period with persistent and volatile in-
ﬂation: this sample is expected to be relatively informative about which models
might be the best forecasting models. Indeed, the MCS consists of relatively
few models, so the MCS proves to be effective at purging the inferior fore-
casts. The later subsample is a period in which inﬂation is relatively smooth
and exhibits little volatility. This yields a sample that contains relatively little
information about which of the models deliver the best forecasts. However,
Stock and Watson (1999) reported that a no-change forecast, which uses last
month’s inﬂation rate as the point forecast, is inferior in both subsamples. In
spite of the relatively low degree of information in the more recent subsam-
ple, we are able to conclude that this no-change forecast is indeed inferior to
other forecasts. We come to this conclusion because the Stock and Watson
no-change forecast never ends up in the MCS. Next, we add the no-change
forecast employed by Atkeson and Ohanian (2001) to the comparison. Their
forecast uses the past year’s inﬂation rate as the point prediction rather than
month-over-month inﬂation. This turns out to matter for the second subsam-
ple, because the no-change (year) forecast has the smallest mean square pre-
diction error (MSPE) of all forecasts. This enables us to reconcile Stock and
Watson (1999)w i t hAtkeson and Ohanian (2001) by showing that their differ-
ent deﬁnitions of the benchmark forecast—no-change (month) and no-change
(year), respectively—explain the different conclusions they reach about these
forecasts.
Our second empirical example shows that the MCS approach is a useful tool
for in-sample evaluation of regression models. This example applies the MCS
to choosing from a set of competing (nominal) interest rate rule regressions on
a quarterly U.S. sample that runs from 1979 through 2006. These regressions
fall into the class of interest rate rules promoted byTa y l o r(1993). His (T aylor’s)
rule forms the basis of a class of monetary policy rules that gauge the success of
monetary policy at keeping inﬂation low and the real economy close to trend.
The MCS does not reveal which T aylor rule regressions best describe the actual
U.S. monetary policy; neither does it identify the best policy rule. Rather the
MCS selects the T aylor rule regressions that have the best empirical ﬁt of the
U.S. federal funds rate in this sample period, where the “best ﬁt” is deﬁned by
different likelihood criteria.
The MCS procedure begins with 25 regression models. We include a pure
ﬁrst-order autoregression, AR(1), of the federal funds rate in the initial MCS.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 5

THE MODEL CONFIDENCE SET 457
The remaining 24 models are T aylor rule regressions that contain different
combinations of lagged inﬂation, lags of various deﬁnitions of real economic
activity (i.e., the output gap, the unemployment rate gap, or real marginal cost),
and in some cases the lagged federal funds rate.
It seems that there is limited information in our U.S. sample for the MCS
procedure to narrow the set of T aylor rule regressions. The one exception is
that the MCS only holds regressions that admit the lagged interest rate. This
includes the pure AR(1). The reason is that the time-series properties of the
federal funds rate is well explained by its own lag. Thus, the lagged federal
funds rate appears to dominate lags of inﬂation and the real activity variables
for explaining the current funds rate. There is some solace for advocates of in-
terest rate rules, because under one likelihood criterion, the MCS often tosses
out T aylor rule regression lacking in lags of inﬂation. Nonetheless, the MCS
indicates that the data are consistent with either lags of the output gap, the un-
employment rate gap, or real marginal cost playing the role of the real activity
variables in the T aylor rule regression. This is not a surprising result. Mea-
surement of gap and marginal cost variables remain an unresolved issue for
macroeconometrics; for example, see Orphanides and V an Norden(2002)a n d
Staiger, Stock, and Watson (1997a). It is also true that monetary policymakers
rely on sophisticated information sets that cannot be spanned by a few aggre-
gate variables (see Bernanke and Boivin (2003)). The upshot is that the sam-
ple used to calculate the MCS has difﬁculties extracting useful information to
separate the pure AR(1) from T aylor rule regressions that include the lagged
federal funds rate.
1.4. Outline of Paper
The paper is organized as follows. We present the theoretical framework of
the MCS in Section 2. Section 3 outlines practical bootstrap methods to imple-
ment the MCS. Multiple model comparison methods related to the MCS are
discussed in Section 4. Section 5 reports the results of simulation experiments.
The MCS is applied to two empirical examples in Section 6. Section 7 con-
cludes. The Supplemental Material ( Hansen, Lunde, and Nason (2011)) pro-
vides detailed description of our bootstrap implementation and some tables
that substantiate the results presented in the simulation and empirical section.
2.
GENERAL THEOR Y FOR MODEL CONFIDENCE SET
In this section, we discuss the theory of model conﬁdence sets for a general
set of alternatives. Our leading example concerns the comparison of empiri-
cal models, such as forecasting models. Nevertheless, we do not make speciﬁc
references to models in the ﬁrst part of this section, in which we lay out the
general theory.
We consider a set, M0/commaorithat contains a ﬁnite number of objects that are
indexed by i = 1/commaori/periodori/periodori/periodori/commaorim0/periodoriThe objects are evaluated in terms of a loss func-
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 6

458 P . R. HANSEN, A. LUNDE, AND J. M. NASON
tion and we denote the loss that is associated with object i in period t as Li/commaorit/commaori
t = 1/commaori/periodori/periodori/periodori/commaorin/periodoriFor example, in the situation where a point forecast ˆYi/commaoritof Yt is
evaluated in terms of a loss function L/commaoriwe deﬁne Li/commaorit=L(Yt/commaoriˆYi/commaorit).
Deﬁne the relative performance variables
dij/commaorit≡ Li/commaorit− Lj/commaorit/commaorifor all i/commaorij∈ M0/periodori
This paper assumes that μij ≡ E(dij/commaorit) is ﬁnite and does not depend on t for all
i/commaorij∈ M0/periodoriWe rank alternatives in terms of expected loss, so that alternative i
is preferred to alternative j if μij < 0/periodori
DEFINITION 1: The set of superior objects is deﬁned by
M∗ ≡{ i ∈ M0 :μij ≤ 0f o ra l lj ∈ M0}/periodori
The objective of the MCS procedure is to determine M∗/periodoriThis is done
through a sequence of signiﬁcance tests, where objects that are found to be
signiﬁcantly inferior to other elements of M0 are eliminated. The hypotheses
that are being tested take the form
H0/commaoriM :μij = 0f o r a l li/commaorij∈ M/commaori(1)
where M ⊂ M0/periodoriWe denote the alternative hypothesis, μij ̸= 0f o rs o m ei/commaorij∈
M/commaoriby HA/commaoriM. Note that H0/commaoriM∗ is true given our deﬁnition of M∗,w h e r e a s
H0/commaoriM is false if M contains elements from M∗ and its complement, M0 \ M∗/periodori
Naturally, the MCS is speciﬁc to a set of candidate models,M0/commaoriand therefore
silent about the relative merits of objects that are not included in M0/periodori
We deﬁne a model conﬁdence set to be any subset of M0 that contains all
of M∗ with a given probability (its coverage probability). The challenge is to
design a procedure that produces a set with the proper coverage probability.
The next subsection introduces a generic MCS procedure that meets this re-
quirement. This MCS procedure is constructed from an equivalence test and
an elimination rule that are assumed to have certain properties. Next, Sec-
tion 3 presents feasible tests and elimination rules that can be used for speciﬁc
problems, such as comparing out-of-sample forecasts and in-sample regression
models.
2.1. The MCS Algorithm and Its Properties
As stated in the Introduction, the MCS procedure is based on anequivalence
test, δ
M/commaoriand an elimination rule, eM/periodoriThe equivalence test, δM/commaoriis used to test
the hypothesis H0/commaoriM for any M ⊂ M0,a n deM identiﬁes the object of M that
is to be removed from M in the event that H0/commaoriM is rejected. As a convention,
we let δM = 0a n dδM = 1 correspond to the cases where H0/commaoriM are accepted
and rejected, respectively.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 7

THE MODEL CONFIDENCE SET 459
DEFINITION 2—MCS Algorithm:
Step 0. Initially set M = M0.
Step 1. T estH0/commaoriM using δM at level α/periodori
Step 2. If H0/commaoriM is accepted, deﬁne ˆM∗
1−α= M; otherwise, use eM to elimi-
nate an object from M and repeat the procedure from Step 1.
The set ˆM∗
1−α
, which consists of the set of surviving objects (those that sur-
vived all tests without being eliminated), is referred to as the model conﬁdence
set.T h e o r e m1, which is stated below, shows that the term “conﬁdence set” is
appropriate in this context, provided that the equivalence test and the elimina-
tion rule satisfy the following assumption.
ASSUMPTION 1: For any M ⊂ M0, we assume about (δM/commaorieM) that
(a) lim supn→∞ P(δM = 1|H0/commaoriM) ≤ α, (b) lim n→∞ P(δM = 1|HA/commaoriM) = 1, and
(c) limn→∞ P(eM ∈ M∗|HA/commaoriM) = 0/periodori
The conditions that Assumption 1 states for δM are standard requirements
for hypothesis tests. Assumption 1(a) requires the asymptotic level not exceed
αand Assumption 1(b) requires the asymptotic power be 1, whereas Assump-
tion 1(c) requires that a superior objecti∗ ∈ M∗ not be eliminated (asn →∞ )
as long as there are inferior models in M.
THEOREM 1—Properties of MCS: Given Assumption 1, it holds that
(i) lim infn→∞ P(M∗ ⊂ ˆM∗
1−α) ≥ 1 − αand (ii) limn→∞ P(i ∈ ˆM∗
1−α
) = 0 for
all i/∈ M∗/periodori
PROOF:L e t i∗ ∈ M∗/periodoriT o prove (i) we consider the event that i∗ is elim-
inated from M/periodoriFrom Assumption 1(c) it follows that P(δM = 1/commaorieM =
i∗|HA/commaoriM) ≤ P(eM =i∗|HA/commaoriM) → 0a s n →∞ /periodoriSo the probability that a good
model is eliminated when M contains poor models vanishes as n →∞ /periodori
Next, Assumption 1(a) shows that lim sup n→∞ P(δM = 1/commaorieM = i∗|H0/commaoriM) =
lim supn→∞ P(δM = 1|H0/commaoriM) ≤ αsuch that the probability that i∗ is eliminated
when all models in M are good models is asymptotically bounded by α/periodoriTo
prove (ii), we ﬁrst note that limn→∞ P(eM =i∗|HA/commaoriM) = 0 such that only poor
models will be eliminated (asymptotically) as long as M ⊈ M∗/periodoriOn the other
hand, Assumption 1(b) ensures that models will be eliminated as long as the
null hypothesis is false. Q.E.D.
Consider ﬁrst the situation where the data contain little information such
that the equivalence test lacks power and the elimination rule may question
a superior model prior to the elimination of all inferior models. The lack of
power causes the procedure to terminate too early (on average), and the MCS
will contain a large number of models, including several inferior models. We
view this as a strength of the MCS procedure. Since lack of power is tied to
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 8

460 P . R. HANSEN, A. LUNDE, AND J. M. NASON
the lack of information in the data, the MCS should be large when there is
insufﬁcient information to distinguish good and bad models.
In the situation where the data are informative, the equivalence test is pow-
erful and will reject all false hypotheses. Moreover, the elimination rule will
not question any superior model until all inferior models have been eliminated.
(This situation is guaranteed asymptotically.) The result is that the ﬁrst time a
superior model is questioned by the elimination rule is when the equivalence
test is applied to
M∗/periodoriThus, the probability that one (or more) superior model
is eliminated is bounded (asymptotically) by the size of the test! Note that ad-
ditional superior models may be eliminated in subsequent tests, but these tests
will only be performed if H0/commaoriM∗ is rejected. Thus, the asymptotic familywise
error rate (FWE), which is the probability of making one or more false rejec-
tions, is bounded by the level that is used in all tests.
Sequential testing is key for building a MCS. However, econometricians of-
ten worry about the properties of a sequential testing procedure, because it
can “accumulate” T ype I errors with unfortunate consequences (see, e.g.,Leeb
and Pötscher (2003)). The MCS procedure does not suffer from this problem
because the sequential testing is halted when the ﬁrst hypothesis is accepted.
When there is only a single model in M∗ (one best model), we obtain a
stronger result.
COROLLAR Y1: Suppose that Assumption 1 holds and that M∗ is a singleton .
Then limn→∞ P(M∗ = ˆM∗
1−α) = 1/periodori
PROOF:W h e n M∗ is a singleton, M∗ ={ i∗}/commaorithen it follows from Theo-
rem 1 that i∗ will be the last surviving element with probability approaching 1
as n →∞ /periodoriThe result now follows, because the last surviving element is never
eliminated. Q.E.D.
2.2. Coherency Between T est and Elimination Rule
The previous asymptotic results do not rely on any direct connection be-
tween the hypothesis test,δM, and the elimination rule,eM. Nonetheless when
the MCS is implemented in ﬁnite samples, there is an advantage to the hypoth-
esis test and elimination rule being coherent. The next theorem establishes a
ﬁnite sample version of the result in Theorem 1(i) when there is a certain co-
herency between the hypothesis test and the elimination rule.
THEOREM 2: Suppose that P(δM = 1/commaorieM ∈ M∗) ≤ α. Then we have
P(M∗ ⊂ ˆM∗
1−α) ≥ 1 − α/periodori
PROOF: We only need to consider the ﬁrst instance that eM ∈ M∗, because
all preceding tests will not eliminate elements that are in M∗/periodoriRegardless of
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 9

THE MODEL CONFIDENCE SET 461
the null hypothesis being true or false, we haveP(δM = 1/commaorieM ∈ M∗) ≤ α/periodoriSo it
follows that αbounds the probability that an element from M∗ is eliminated.
Additional elements from M∗ may be eliminated in subsequent tests, but these
test will only be undertaken if all preceding tests are rejected. So we conclude
that P(M∗ ⊂ ˆM∗
1−α) ≥ 1 − α. Q.E.D.
The property that P(δM = 1/commaorieM ∈ M∗) ≤ αholds under both the null hy-
pothesis and the alternative hypothesis is key for the result in Theorem 2.
For a test with the correct size, we have P(δM = 1|H0/commaoriM) ≤ α/commaoriwhich implies
P(δM = 1/commaorieM ∈ M∗|H0/commaoriM) ≤ α/periodoriThe additional condition, P(δM = 1/commaorieM ∈
M∗|HA/commaoriM) ≤ α/commaoriensures that a rejection, δM = 1, can be taken as signiﬁcant
evidence that eM is not in M∗.
In practice, hypothesis tests often rely on asymptotic results that cannot
guarantee P(δM = 1/commaorieM ∈ M∗) ≤ αholds in ﬁnite samples. We provide a
deﬁnition of coherency between a test and an elimination rule that is useful
in situations where testing is grounded on asymptotic distributions. In what
follows, we use P0 to denote the probability measure that arises via imposing
the null hypothesis via the transformation dij/commaorit↦→ dij/commaorit− μij/periodoriThus P is the true
probability measure, whereas P0 is a simple transformation of P that satisﬁes
the null hypothesis.
DEFINITION 3: There is said to be coherency between test and elimination
rule when
P(δM = 1/commaorieM ∈ M∗) ≤ P0(δM = 1)/periodori
The coherency in conjunction with an asymptotic control of the T ype I er-
ror, lim supn→∞ P0(δM = 1) ≤ α/commaoritranslates into an asymptotic version of the
assumption we made in Theorem 2. Coherency places restrictions on the com-
binations of tests and elimination rules we can employ. These restrictions go
beyond those imposed by the asymptotic conditions we formulated in Assump-
tion 1. In fact, coherency serves to curb the reliance on asymptotic properties
so as to avoid perverse outcomes in ﬁnite samples that could result from absurd
combinations of test and elimination rule. Coherency prevents us from adopt-
ing the most powerful test of the hypothesis H
0/commaoriM in some situations. The rea-
son is that tests do not necessarily identify a single element as the cause for the
rejection. A good analogy is found in the standard regression model, where an
F-test may reject the joint hypothesis that all regression coefﬁcients are zero,
even though all t-statistics are insigniﬁcant.2
In our bootstrap implementations of the MCS procedure, we adopt the re-
quired coherency between the test and the elimination rule.
2Another analogy is that it is easier to conclude that a murder has taken place than it is to
determine who committed the murder.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 10

462 P . R. HANSEN, A. LUNDE, AND J. M. NASON
2.3. MCS p-V alues
In this section we introduce the notion of MCS p-values. The elimination
rule, eM/commaorideﬁnes a sequence of (random) sets M0 = M1 ⊃ M2 ⊃···⊃ Mm0/commaori
where Mi ={ eMi/commaori/periodori/periodori/periodori/commaorieMm0
} and m0 is the number of elements in M0/periodoriSo
eM0 = eM1 is the ﬁrst element to be eliminated in the event that H0/commaoriM1/commaoriis
rejected, eM2 is the second element to be eliminated, and so forth.
DEFINITION 4—MCS p-V alues: LetPH0/commaoriMi
denote the p-value associated
with the null hypothesisH0/commaoriMi/commaoriwith the convention thatPH0/commaoriMm0
≡ 1/periodoriThe MCS
p-value for model eMj ∈ M0 is deﬁned by ˆpeMj
≡ maxi≤j PH0/commaoriMi
.
The advantage of this deﬁnition of MCS p- v a l u e sw i l lb ee v i d e n tf r o mT h e -
orem 3 which is stated below. Since Mm0 consists of a single model, the null
hypothesis, H0/commaoriMm0
/commaorisimply states that the last surviving model is as good as
itself, making the convention PH0/commaoriMm0
≡ 1l o g i c a l .
Ta b l eI illustrates how MCS p-values are computed and how they relate to
p-values of the individual tests PH0/commaoriMi
/commaorii= 1/commaori/periodori/periodori/periodori/commaorim0.T h eM C Sp-values are
convenient because they make it easy to determine whether a particular object
is in ˆM∗
1−α for any α/periodoriThus, the MCS p-values are an effective way to convey
the information in the data.
THEOREM 3: Let the elements of M0 be indexed by i = 1/commaori/periodori/periodori/periodori/commaorim0/periodoriThe MCS
p-value, ˆpi/commaoriis such that i ∈ ˆM∗
1−α
if and only if ˆpi ≥ αfor any i ∈ M0/periodori
TABLE I
COMPUTATION OF MCS p-VALUESa
Elimination Rule p-V alue forH0/commaoriMk MCS p-Value
eM1 PH0/commaoriM1
= 0/periodori01 ˆpeM1
= 0/periodori01
eM2 PH0/commaoriM2
= 0/periodori04 ˆpeM2
= 0/periodori04
eM3 PH0/commaoriM3
= 0/periodori02 ˆpeM3
= 0/periodori04
eM4 PH0/commaoriM4
= 0/periodori03 ˆpeM4
= 0/periodori04
eM5 PH0/commaoriM5
= 0/periodori07 ˆpeM5
= 0/periodori07
eM6 PH0/commaoriM6
= 0/periodori04 ˆpeM6
= 0/periodori07
eM7 PH0/commaoriM7
= 0/periodori11 ˆpeM7
= 0/periodori11
eM8 PH0/commaoriM8
= 0/periodori25 ˆpeM8
= 0/periodori25
/periodori/periodori/periodori
/periodori/periodori/periodori
/periodori/periodori/periodori
eM(m0) PH0/commaoriMm0
≡ 1/periodori00 ˆpeMm0
= 1/periodori00
aNote that MCS p-values for some models do not coincide with the p-values for
the corresponding null hypotheses. For example, the MCSp-value for eM3 (the third
model to be eliminated) exceeds the p-value for H0/commaoriM3 , because the p-value associ-
ated with H0/commaoriM2 —a null hypothesis tested prior to H0/commaoriM3 —is larger.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 11

THE MODEL CONFIDENCE SET 463
PROOF: Suppose that ˆpi <αand determine the k for which i =eMk/periodoriSince
ˆpi = ˆpeMk
= maxj≤k PH0/commaoriMj
, it follows that H0/commaoriM1/commaori/periodori/periodori/periodori/commaoriH0/commaoriMk are all rejected at
signiﬁcance level α. Hence, the ﬁrst accepted hypothesis (if any) occurs after
i =eMk has been eliminated. So ˆpi <α implies i/∈ ˆM∗
1−α/periodoriSuppose now that
ˆpi ≥ α/periodoriThen for some j ≤ k,w eh a v ePH0/commaoriMj
≥ α/commaoriin which case H0/commaoriMj is ac-
cepted at signiﬁcance level αthat terminates the MCS procedure before the
elimination rule gets to eMk =i/periodoriSo ˆpi ≥ αimplies i ∈ ˆM∗
1−α/periodoriThis completes
the proof. Q.E.D.
The interpretation of a MCS p-value is analogous to that of a classical p-
value. The analogy is to a (1 − α)conﬁdence interval that contains the “true”
parameter with a probability no less than 1− α.T h eM C Sp-value also cannot
be interpreted as the probability that a particular model is the best model,
exactly as a classical p-value is not the probability that the null hypothesis is
true. Rather, the probability interpretation of a MCS p-value is tied to the
random nature of the MCS because the MCS is a random subset of models
that contains M∗ with a certain probability.
3. BOOTSTRAP IMPLEMENTATION
3.1. Equivalence T ests and Elimination Rules
Now we consider speciﬁc equivalence tests and an elimination rule that sat-
isfy Assumption 1. The following assumption is sufﬁciently strong to enable us
to implement the MCS procedure with bootstrap methods.
ASSUMPTION 2: For some r> 2 and γ>0, it holds that E|dij/commaorit|r+γ< ∞ for all
i/commaorij∈ M0 and that {dij/commaorit}i/commaorij∈M0 is strictly stationary with var(dij/commaorit)> 0 andα-mixing
of order −r/(r − 2).
Assumption 2 places restrictions on the relative performance variables,
{dij/commaorit}/commaorinot directly on the loss variables {Li/commaorit}/periodoriFor example, a loss function
need not be stationary as long as the loss differentials, {dij/commaorit}/commaorii /commaorij∈ M0/commaorisatisfy
Assumption 2. The assumption allows for some types of structural breaks and
other features that can create nonstationary {Li/commaorit} as long as all objects in M0
are affected in a similar way that preserves the stationarity of {dij/commaorit}/periodori
3.1.1. Quadratic-Form T est
Let M be some subset of M0 and let m be the number of models in M =
{i1/commaori/periodori/periodori/periodori/commaoriim}. We deﬁne the vector of loss variables Lt ≡ (Li1/commaorit/commaori/periodori/periodori/periodori/commaoriLim/commaorit)′/commaorit=
1/commaori/periodori/periodori/periodori/commaorin/commaoriand its sample average ¯L ≡ n−1 ∑n
t=1 Lt/commaoriand we let ι≡ (1/commaori/periodori/periodori/periodori/commaori1)′ be
the column vector where allm entries equal 1. The orthogonal complement to
ιis an m ×(m − 1) matrix, ι⊥ that has full column rank and satisﬁes ι′
⊥ι= 0
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 12

464 P . R. HANSEN, A. LUNDE, AND J. M. NASON
(a vector of zeros). The m − 1-dimensional vector Xt ≡ ι′
⊥Lt can be viewed as
m − 1 contrasts, because each element of Xt is a linear combination of dij/commaorit,
i/commaorij∈ M/commaoriwhich has mean zero under the null hypothesis.
LEMMA 1: Given Assumption 2, letXt ≡ ι′
⊥
Lt and deﬁne θ≡ E(Xt). The null
hypothesis H0/commaoriM is equivalent to θ= 0 and it holds that n1/2( ¯X − θ)
d
→ N(0/commaoriΣ)/commaori
where ¯X ≡ n−1 ∑n
t=1 Xt and Σ≡ limn→∞ var(n1/2 ¯X)/periodori
PROOF: Note that Xt =ι′
⊥Lt can be written as a linear combination of dij/commaorit,
i/commaorij∈ M0/commaoribecause ι′
⊥ι= 0/periodoriThus H0/commaoriM is given by θ= 0 and the asymptotic
normality follows by the central limit theorem forα-mixing processes (see, e.g.,
White (2000a)). Q.E.D.
Lemma 1 shows that H0/commaoriM can be tested using traditional quadratic-form
statistics. An example is TQ ≡ n ¯X′ ˆΣ# ¯X,w h e r eˆΣis some consistent estimator
of Σand ˆΣ# denotes the Moore–Penrose inverse of ˆΣ.3 The rank q ≡ rank( ˆΣ)
represents the effective number of contrasts (the number of linearly indepen-
dent comparisons) under H0/commaoriM.S i n c eˆΣ
p
→ Σ(by assumption), it follows that
TQ
d
→ χ2
(q)
,w h e r eχ2
(q)
denotes the χ2 distribution with q degrees of freedom.
Under the alternative hypothesis, TQ diverge to inﬁnity with probability 1. So
the testδM will meet the requirements of Assumption1 when constructed from
TQ/periodoriAlthough the matrix ι⊥ is not fully identiﬁed by the requirements ι′
⊥
ι= 0
and det(ι′
⊥ι⊥) ̸= 0 (but the subspace spanned by the columns of ι⊥ is), there is
no problem because the statistic TQ is invariant to the choice for ι⊥/periodori
A rejection of the null hypothesis based on the quadratic-form test need not
identify an inferior alternative because a large value of TQ can stem from sev-
eral ¯dij being slightly different from zero. T o achieve the required coherence
between test and elimination rule, additional testing is needed. Speciﬁcally,
one needs to test all subhypotheses of any rejected hypothesis, unless the sub-
hypothesis is nested in an accepted hypothesis, before further elimination is
justiﬁed. The underlying principle is known as the closed testing procedure (see
Lehmann and Romano (2005, pp. 366–367)).
When m is large relative to the sample size, n/commaorireliable estimates of Σare
difﬁcult to obtain, because the number of elements ofΣto be estimated are of
orderm
2/periodoriIt is convenient to use a test statistic that does not require an explicit
estimate of Σin this case. We consider test statistics that resolve this issue in
the next section.
3Under the additional assumption that {dij/commaorit}i/commaorij∈M is uncorrelated (across t), we can use
ˆΣ=n−1 ∑n
t=1(Xt − ¯X)(Xt − ¯X)′. Otherwise, we need a robust estimator along the lines ofNewey
and West (1987). In the context of comparing forecasts, West and Cho (1995) were the ﬁrst in-
vestigators to use the test statistic TQ. They based their test on (asymptotic) critical values from
χ2
(m−1).
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 13

THE MODEL CONFIDENCE SET 465
3.1.2. T ests Constructed Fromt-Statistics
This section develops two tests that are based on multiple t-statistics. This
approach has two advantages. First, it bypasses the need for an explicit esti-
mate ofΣ/periodoriSecond, the multiplet-statistic approach simpliﬁes the construction
of an elimination rule that satisﬁes the notion of coherency formulated in De-
ﬁnition 3.
Deﬁne the relative sample loss statistics ¯dij ≡ n−1 ∑n
t=1 dij/commaoritand ¯di· ≡
m−1 ∑
j∈M
¯dij/periodoriHere ¯dij measures the relative sample loss between the ith and
jth models, while ¯di· is the sample loss of the ith model relative to the average
across models in M/periodoriThe latter can be seen from the identity ¯di· =( ¯Li − ¯L·)/commaori
where ¯Li ≡ n−1 ∑n
t=1 Li/commaoritand ¯L· ≡ m−1 ∑
i∈M
¯Li/periodoriFrom these statistics, we con-
struct the t-statistics
tij =
¯dij
√
ˆvar( ¯dij)
and ti· =
¯di·
√
ˆvar( ¯di·)
for i/commaorij∈ M/commaori
where ˆvar( ¯dij) and ˆvar( ¯di·) denote estimates of var ( ¯dij) and var( ¯di·), respec-
tively. The ﬁrst statistic, tij/commaoriis used in the well known test for comparing two
forecasts; see Diebold and Mariano (1995)a n dWest (1996). The t-statistics tij
and ti· are associated with the null hypothesis that Hij :μij = 0a n dHi· :μi· = 0,
where μi· = E( ¯di·)/periodoriThese statistics form the basis of tests of the hypothe-
sis H0/commaoriM. We take advantages of the equivalence between H0/commaoriM/commaori{Hij for all
i/commaorij∈ M},a n d{Hi· for all i ∈ M}.W i t hM ={i1/commaori/periodori/periodori/periodori/commaoriim} the equivalence fol-
lows from
μi1 =···= μim ⇔ μij = 0f o r a l l i/commaorij∈ M
⇔ μi· = 0f o r a l l i ∈ M/periodori
Moreover, the equivalence extends to{μi· ≤ 0f o ra l li ∈ M} as well as{|μij|≤ 0
for all i/commaorij∈ M}/commaoriand these two formulations of the null hypothesis map natu-
rally into the test statistics
Tmax/commaoriM = max
i∈M
ti· and TR/commaoriM ≡ max
i/commaorij∈M
|tij|/commaori
which are available to test the hypothesis H0/commaoriM.4 The asymptotic distributions
of these test statistics are nonstandard because they depend on nuisance pa-
rameters (under the null and the alternative). However, the nuisance para-
meters pose few obstacles, as the relevant distributions can be estimated with
bootstrap methods that implicitly deal with the nuisance parameter problem.
4An earlier version of this paper has results for the test statistics TD = ∑n
j=1 t2
i· and TQ/periodori
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 14

466 P . R. HANSEN, A. LUNDE, AND J. M. NASON
This feature of the bootstrap has previously been used in this context byKilian
(1999), White (2000b), Hansen (2003b, 2005 ), and Clark and McCracken
(2005).
Characterization of the MCS procedure needs an elimination rule, eM/commaori
that meets the requirements of Assumption 1(c) and the coherency of Deﬁ-
nition 3. For the test statistic Tmax/commaoriM, the natural elimination rule is emax/commaoriM ≡
arg maxi∈M ti· because a rejection of the null hypothesis identiﬁes the hy-
pothesis μj· = 0 as false for j = emax/commaoriM/periodoriIn this case the elimination rule re-
moves the model that contributes most to the test statistic. This model has
the largest standardized excess loss relative to the average across all mod-
els in
M/periodoriWith the other test statistic, TR/commaoriM/commaorithe natural elimination rule is
eR/commaoriM = arg maxi∈M supj∈M tij because this model is such that teR/commaoriMj =TR/commaoriM for
some j ∈ M/periodoriThese combinations of test and elimination rule will satisfy the
required coherency.
PROPOSITION 1: Let δmax/commaoriM and δR/commaoriM denote the tests based on the statistics
Tmax/commaoriM and TR/commaoriM/commaorirespectively. Then (δmax/commaoriM/commaoriemax/commaoriM) and (δR/commaoriM/commaorieR/commaoriM) satisfy
the coherency of Deﬁnition 3.
PROOF:L e t Ti denote eitherti· or maxj∈M tij/commaoriand note that the test statistics
Tmax/commaoriM and TR/commaoriM are both of the form T = maxi∈M Ti/periodoriLet P0 be as deﬁned in
Section 2.2. From the deﬁnitions of ti· and tij,w eh a v ef o ri ∈ M∗ the ﬁrst-
order stochastic dominance result P0(maxi∈M′ Ti >x) ≥ P(maxi∈M′ Ti >x) for
any M′ ⊂ M∗ and all x ∈ R/periodoriThe coherency now follows from
P(T >c/commaorieM =i for somei ∈ M∗)
=P(T >c/commaoriT=Ti for somei ∈ M∗)
=P
(
max
i∈M∩M∗
Ti >c/commaoriTi ≥ Tj for allj ∈ M
)
≤ P
(
max
i∈M∩M∗
Ti >c
)
≤ P0
(
max
i∈M∩M∗
Ti >c
)
≤ P0
(
max
i∈M
Ti >c
)
=P0(T >c)/periodori
This completes the proof. Q.E.D.
Next, we establish two intermediate results that underpin the bootstrap im-
plementation of the MCS.
LEMMA 2: Suppose that Assumption 2 holds and deﬁne ¯Z = ( ¯d1·/commaori/periodori/periodori/periodori/commaori¯dm·)′/periodori
Then
n1/2( ¯Z − ψ)
d
→ Nm(0/commaoriΩ)as n →∞ /commaori(2)
where ψ≡ E( ¯Z) and Ω≡ limn→∞ var(n1/2 ¯Z)/commaoriand the null hypothesis H0/commaoriM is
equivalent to: ψ= 0/periodori
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 15

THE MODEL CONFIDENCE SET 467
PROOF: From the identity ¯di· = ¯Li − ¯L· = ¯Li − m−1 ∑
j∈M
¯Lj = m−1 ×∑
j∈M( ¯Li − ¯Lj) =m−1 ∑
j∈M
¯dij, we see that the elements of ¯Z are linear trans-
formations of ¯X from Lemma 1.T h u sf o rs o m e(m− 1)×m matrixG,w eh a v e
¯Z = G′ ¯X and the result now follows, where ψ= G′θand Ω= G′ΣG/periodori(The
m ×m covariance matrix Ωhas reduced rank, as rank(Ω)≤ m − 1.) Q.E.D.
In the following discussion, we let ϱ denote the m ×m correlation matrix
that is implied by the covariance matrix Ω of Lemma 2. Further, given the
vector of random variables ξ∼ Nm(0/commaoriϱ)/commaoriwe let Fϱ denote the distribution of
maxi ξi.
THEOREM 4: Let Assumption 2 hold and suppose that ˆω2
i ≡ ˆvar(n1/2 ¯di·) =
n ˆvar( ¯di·)
p
→ ω2
i
/commaoriwhere ω2
i
, i = 1/commaori/periodori/periodori/periodori/commaorim, are the diagonal elements of Ω/periodoriUnder
H0/commaoriM, we have Tmax/commaoriM
d
→ Fϱ, and under the alternative hypothesis HA/commaoriM/commaoriwe have
Tmax/commaoriM →∞ in probability. Moreover, under the alternative hypothesis , we have
Tmax/commaoriM =tj·, where j =emax/commaoriM /∈ M∗ for n sufﬁciently large.
PROOF:L e t D ≡ diag(ω2
1
/commaori/periodori/periodori/periodori/commaoriω2
m
) and ˆD ≡ diag( ˆω2
1
/commaori/periodori/periodori/periodori/commaoriˆω2
m
)/periodoriFrom
Lemma 2 it follows that ξn =(ξ1/commaorin/commaori/periodori/periodori/periodori/commaoriξm/commaorin)′ ≡ D−1/2n1/2 ¯Z
d
→ Nm(0/commaoriϱ),s i n c e
ϱ = D−1/2ΩD−1/2/periodoriFrom ti· = ¯di·/
√
ˆvar( ¯di·) = n1/2 ¯di·/ ˆωi = ξi/commaorin
ωi
ˆωi
,i tn o wf o l -
lows that Tmax/commaoriM = maxi ti· = maxi( ˆD−1/2n1/2 ¯Z)i
d
→ Fϱ/periodoriUnder the alterna-
tive hypothesis, we have ¯dj·
p
→ μj· > 0f o ra n yj/∈ M∗/commaoriso that both tj· and
Tmax/commaoriM diverge to inﬁnity at rate n1/2 in probability. Moreover, it follows that
emax/commaoriM /∈ M∗ for n sufﬁciently large. Q.E.D.
Theorem 4 shows that the asymptotic distribution of Tmax/commaoriM depends on the
correlation matrixϱ/periodoriNonetheless, as discussed earlier, bootstrap methods can
be employed to deal with this nuisance parameter problem. Thus, we con-
struct a test ofH0/commaoriM by comparing the test statisticTmax/commaoriM to an estimate of the
95% quantile, say, of its limit distribution under the null hypothesis. Although
the quantile may depend on ϱ/commaoriour bootstrap implementation leads to an as-
ymptotically valid test because the bootstrap consistently estimates the desired
quantile. A detailed description of our bootstrap implementation is available
in a separate appendix (H a n s e n ,L u n d e ,a n dN a s o n(2011)).
Theorem 4 formulates results for the situation where the MCS is constructed
with T
max/commaoriM and emax/commaoriM = arg maxi ti·/periodoriSimilar results hold for the MCS that is
constructed from TR/commaoriM and eR/commaoriM/periodoriThe arguments are almost identical to those
used for Theorem 4.
3.2. MCS for Regression Models
This section shows how to construct the MCS for regression models using
likelihood-based criteria. Information criteria, such as the AIC and BIC, are
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 16

468 P . R. HANSEN, A. LUNDE, AND J. M. NASON
special cases for building a MCS of regression models. The MCS approach de-
parts from standard practice where the AIC and BIC select a single model, but
are silent about the uncertainty associated with this selection.
5 Thus, the MCS
procedure yields valuable additional information about the uncertainty sur-
rounding model selection. In Section 6.2, application of the MCS procedure in
sample to T aylor rule regressions indicates this uncertainty can be substantial.
Although we focus on regression models for simplicity, it will be evident that
the MCS procedure laid out in this setting can be adapted to more complex
models, such as the type of models analyzed in Sin and White (1996).
3.2.1. Framework and Assumptions
Consider the family of regression models Yt = β′
jXj/commaorit+ εj/commaorit, t = 1/commaori/periodori/periodori/periodori/commaorin,
where Xj/commaoritis a subset of the variables in Xt for j = 1/commaori/periodori/periodori/periodori/commaorim0/periodoriThe set of re-
gression models, M0/commaorimay consist of nested, nonnested, and overlapping spec-
iﬁcations.
Throughout we assume that the pair (Yt/commaoriX′
t) is strictly stationary and sat-
isﬁes Assumption 1 in Goncalves and White (2005). This justiﬁes our use
of the moving-block bootstrap to implement our resampling procedure. The
framework of Goncalves and White (2005) permits weak serial dependence in
(Yt/commaoriX′
t)/commaoriwhich is important for many applications.
The population parameters for each of the models are deﬁned by β0j =
[E(Xj/commaoritX′
j/commaorit)]−1E(Xj/commaoritYt) and σ2
0j = E(ε2
j/commaorit)/commaoriwhere εj/commaorit= Yt − β′
0jXj/commaorit/commaorit=
1/commaori/periodori/periodori/periodori/commaorin/periodoriFurthermore, the Gaussian quasi-log-likelihood function is, apart
from a constant, given by
ℓ(βj/commaoriσ2
j ) =− n
2 logσ2
j − σ−2
j
1
2
n∑
t=1
(Yt − β′
jXj/commaorit)2/periodori
3.2.2. MCS by Kullback–Leibler Divergence
One way to deﬁne the best regression model is in terms of the Kullback–
Leibler information criterion (KLIC) (see, e.g., Sin and White (1996)). This is
equivalent to ranking the models in terms of the expected value of the quasi-
log-likelihood function when evaluated at their respective population parame-
ters, that is, E[ℓ(β
0j/commaoriσ2
0j)]/periodoriIt is convenient to deﬁne
Q(Z/commaoriθj) =− 2ℓ(βj/commaoriσ2
j ) =n logσ2
j +
n∑
t=1
(Yt − β′
jXj/commaorit)2
σ2
j
/commaori
5The same point applies to the Autometrics procedure; see Doornik (2009) and references
therein. Autometrics is constructed from a collection of tests and decision rules but does not
control a familywise error rate, and the set of models that Autometrics seeks to identify is not
deﬁned from a single criterion, such as the Kullback–Leibler information criterion.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 17

THE MODEL CONFIDENCE SET 469
where θj can be viewed as a high-dimensional vector that is restricted by the
parameter spaceΘj ⊂ Θthat deﬁnes thejth regression model. The population
parameters are here given by θ0j = arg minθ∈Θj E[Q(Z/commaoriθ)], j = 1/commaori/periodori/periodori/periodori/commaorim0/commaoriand
the best model is deﬁned by min j E[Q(Z/commaoriθ0j)]/periodoriIn the notation of the MCS
framework, the KLIC leads to
M∗
KLIC =
{
j :E[Q(Z/commaoriθ0j)]= min
i
E[Q(Z/commaoriθ0i)]
}
/commaori
which (as always) permits the existence of more than one best model. 6 The
extension to other criteria, such as the AIC and the BIC, is straightforward.
For instance, the set of best models in terms of the AIC is given by M∗
AIC =
{j :E[Q(Z/commaoriθ0j) + 2kj]= mini E[Q(Z/commaoriθ0i) + 2ki]},w h e r ekj is the degrees of
freedom in the jth model.
The likelihood framework enables us to construct either ˆM∗
KLIC
or ˆM∗
AIC
by drawing on the theory of quasi-maximum-likelihood estimation (see,
e.g., White (1994)). Since the family of regression models is linear, the
quasi-maximum-likelihood estimators are standard, ˆβj = (∑n
t=1 Xj/commaoritX′
j/commaorit)−1 ×∑n
t=1
Xj/commaoritYt/commaoriand ˆσ2
j =n−1 ∑n
t=1
ˆε2
j/commaorit/commaoriwhere ˆεj/commaorit=Yt − ˆβ′
jXj/commaorit/periodoriWe have
Q(Z/commaoriˆθj) − Q(Z/commaoriθ0j)
=n
{
(logσ2
0j − log ˆσ2
j ) +
(
n−1
n∑
t=1
ε2
j/commaorit/σ2
0j − 1
)}
/commaori
which is the quasi-likelihood ratio (QLR) statistic for the null hypothesis,
H0 :θ=θ0j .
In the event that the jth model is correctly speciﬁed, it is well known that
the limit distribution of Q(Z/commaoriˆθj) − Q(Z/commaoriθ0j) is χ2
(kj)/commaoriwhere the degrees of
freedom, kj/commaoriis given by the dimension of θ0j =(β′
0j/commaoriσ2
0j)′/periodoriIn the present mul-
timodel setup, it is unlikely that all models are correctly speciﬁed. More gener-
ally, the limit distribution of the QLR statistic has the form,∑kj
i=1 λi/commaorijZ2
i/commaorij/commaoriwhere
λ1/commaorij/commaori/periodori/periodori/periodori/commaoriλkj/commaorijare the eigenvalues of I−1
j Jj and Z1/commaorij/commaori/periodori/periodori/periodori/commaoriZkj/commaorij∼ i/periodorii/periodorid/periodoriN(0/commaori1).
The information matrices Ij and Jj are those associated with the jth model,
6In the present situation, we have E [Q(Zj/commaoriθ0j)] ∝σ2
0j/periodoriThe implication is that the error vari-
ance, σ2
0j/commaoriinduces the same ranking as KLIC, so that M∗
KLIC ={j :σ2
0j = minj′ σ2
0j′ }/periodori
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 18

470 P . R. HANSEN, A. LUNDE, AND J. M. NASON
Ij = diag(σ−2
0j E(Xj/commaoritX′
j/commaorit)/commaori1
2σ−4
0j ) and
Jj = E
⎛
⎜⎜
⎜
⎜⎝
σ
−4
0j n−1
n∑
s/commaorit=1
Xj/commaorisεj/commaorisεj/commaoritX′
j/commaorit
1
2σ−6
0j n−1
n∑
s/commaorit=1
Xj/commaorisεj/commaorisε2
j/commaorit
• 1
4σ−8
0j n−1
n∑
s/commaorit=1
(ε2
j/commaorisε2
j/commaorit−σ4
0j)
⎞
⎟⎟⎟
⎟⎠
/periodori
The effective degrees of freedom, k
⋆
j/commaoriis deﬁned by the mean of the QLR limit
distribution:
k⋆
j =λ1/commaorij+···+ λkj/commaorij= tr{I−1
j Jj}
= tr
{
[E(Xj/commaoritX′
j/commaorit)]−1σ−2
0j n−1
n∑
s/commaorit=1
E(Xj/commaorisεj/commaorisX′
j/commaoritεj/commaorit)
}
+n−1 1
2
n∑
s/commaorit=1
E
( ε2
j/commaorisε2
j/commaorit
σ4
0j
− 1
)
/periodori
The previous expression points to estimating k⋆
j with heteroskedasticity and
autocorrelation consistent (HAC) type estimators that account for the auto-
correlation in {Xj/commaoritεj/commaorit} and {ε2
j/commaorit} (e.g., Newey and West (1987)a n d Andrews
(1991)). Below we use a simple bootstrap estimate of k⋆
j/commaoriwhich is also em-
ployed in our simulations and our empirical T aylor rule regression application.
The effective degrees of freedom in the context of misspeciﬁed models was
ﬁrst derived by Ta k e u c h i(1976). He proposed a modiﬁed AIC, sometimes re-
ferred to as the T akeuchi information criterion (TIC), which computes the
penalty with the effective degrees of freedom rather than the number of pa-
r a m e t e r sa si su s e db yt h eA I C ;s e ea l s oSin and White (1996)a n dHong and
Preston (2008). We use the notation AIC
⋆ and BIC⋆ to denote the information
criteria that are deﬁned by substituting the effective degrees of freedomk⋆
j for
kj in the AIC and BIC, respectively. In this case, our AIC ⋆ is identical to the
TIC proposed by Ta k e u c h i(1976).
3.2.3. The MCS Procedure
The MCS procedure can be implemented by the moving-block bootstrap
applied to the pair (Yt/commaoriXt);s e eGoncalves and White (2005). We compute re-
samples Z∗
b =(Y∗
b/commaorit/commaoriX∗
b/commaorit)n
t=1 for b = 1/commaori/periodori/periodori/periodori/commaoriB/commaoriwhich equates the original point
estimate, ˆθj , to the population parameter in thejth model under the bootstrap
scheme.
The literature has proposed several bootstrap estimators of the effective
degrees of freedom, k⋆
j = E[Q(Z/commaoriθ0j) − Q(Z/commaoriˆθj)]; see, for example, Efron
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 19

THE MODEL CONFIDENCE SET 471
(1983, 1986)a n dCavanaugh and Shumway (1997). These and additional esti-
mators are analyzed and compared in Shibata (1997). We adopt the estimator
for k⋆
j that is labelled B3 in Shibata (1997). In the regression context, this esti-
mator takes the form
ˆk⋆
j =B−1
B∑
b=1
Q(Z∗
b/commaoriˆθj) − Q(Z∗
b/commaoriˆθ∗
b/commaorij)
=B−1
B∑
b=1
{
n log
ˆσ2
j
ˆσ∗2
b/commaorij
+
n∑
t=1
(ε∗
b/commaorij/commaorit)2
ˆσ2
j
− n
}
/commaori
whereε∗
b/commaorij/commaorit=Y∗
b/commaorit− ˆβ′
jX∗
b/commaorij/commaorit/commaoriˆε∗
b/commaorij/commaorit=Y∗
b/commaorit− ˆβ∗′
b/commaorijX∗
b/commaorij/commaorit/commaoriand ˆσ∗2
b/commaorij=n−1 ∑n
t=1(ˆε∗
b/commaorij/commaorit)2/periodori
This is an estimate of the expected overﬁt that results from maximization of
the likelihood function. For a correctly speciﬁed model, we havek⋆
j =kj ,s ow e
would expect ˆk⋆
j ≈ kj when the jth model is correctly speciﬁed. This is indeed
what we ﬁnd in our simulations; see Section 5.2.
Given an estimate of the effective degrees of freedom ˆk⋆
j/commaoricompute the AIC⋆
statisticQ(Z/commaoriˆθj)+ ˆk⋆
j
, which is centered about E{Q(Z/commaoriθ0j)}/periodoriThe null hypoth-
esis H0/commaoriM states that E[Q(Z/commaoriθ0i) − Q(Z/commaoriθ0j)]= 0f o ra l li/commaorij∈ M/periodoriThis moti-
vates the range statistic
TR/commaoriM = max
i/commaorij∈M
⏐⏐[Q(Z/commaoriˆθi) + ˆk⋆
i ]−[ Q(Z/commaoriˆθj) + ˆk⋆
j
]
⏐⏐
and the elimination rule eM = arg maxj∈M[Q(Z/commaoriˆθj) + ˆk⋆
j ]/periodoriThis elimination
rule removes the model with the largest bias adjusted residual variance. Our
test statistic,TR/commaoriM/commaoriis a range statistic over recentered QLR statistics computed
f o ra l lp a i r so fm o d e l si nM/periodoriIn the special case with independent and identi-
cally distributed (i.i.d.) data and just two models in M/commaoriwe could simply adopt
the QLR test of Vuong (1989) as our equivalence test.
Next, we estimate the distribution of TR/commaoriM under the null hypothesis. The
estimate is calculated with methods similar to those used inWhite (2000b)a n d
Hansen (2005). The joint distribution of
(
Q(Z/commaoriˆθ1) +k⋆
1 − E[Q(Z/commaoriθ01)]/commaori/periodori/periodori/periodori/commaori
Q
(
Z/commaoriˆθm0
)
+k⋆
m
0 − E
[
Q
(
Z/commaoriθ0m0
)])
is estimated by the empirical distribution of
{
Q(Z∗
b/commaoriˆθ∗
b/commaori1) + ˆk⋆
1 − Q(Z/commaoriˆθ1)/commaori/periodori/periodori/periodori/commaoriQ
(
Z∗
b/commaoriˆθ∗
b/commaorim0
)
+ ˆk⋆
m
0 − Q
(
Z/commaoriˆθm0
)}
(3)
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 20

472 P . R. HANSEN, A. LUNDE, AND J. M. NASON
for b = 1/commaori/periodori/periodori/periodori/commaoriB/commaoribecause Q(Z/commaoriˆθj) plays the role of E [Q(Z/commaoriθ0j)] under the
resampling scheme. These bootstrap statistics are relatively easy to compute
because the structure of the likelihood function is
Q(Z∗
b/commaoriˆθ∗
b/commaorij) − Q(Z/commaoriˆθj) =n(log ˆσ∗2
b/commaorij+ 1) − n(log ˆσ2
j + 1) =n log
ˆσ∗2
b/commaorij
ˆσ2
j
/commaori
where ˆσ∗2
b/commaorij=n−1 ∑n
t=1(Y∗
b/commaorit− ˆβ∗′
b/commaorijX∗
b/commaorij/commaorit)2/periodoriFor each of the bootstrap resamples,
we compute the test statistic
T∗
b/commaoriR/commaoriM = max
i/commaorij∈M
⏐⏐{Q(Z∗
b/commaoriˆθ∗
b/commaorii) + ˆk⋆
i − Q(Z/commaoriˆθi)}
−{ Q(Z∗
b/commaoriˆθ∗
b/commaorij) + ˆk⋆
j − Q(Z/commaoriˆθj)}
⏐⏐/periodori
The p-value for the hypothesis test with which we are concerned is computed
by
pM =B−1
B∑
b=1
1{T∗
b/commaoriR/commaoriM≥TR/commaoriM}/periodori
The empirical distribution of n−1/2T∗
b/commaoriR/commaoriM yields a conservative estimate of the
distribution of n−1/2TR/commaoriM as n/commaoriB→∞ /periodoriT h ec o n s e r v a t i v en a t u r eo ft h i se s t i -
mate refers to the p-value, pM/commaoribeing conservative in situations where the
comparisons involve nested models. We discuss this issue at some length in the
next subsection.
It is also straightforward to construct the MCS using either the AIC, the
BIC, the AIC⋆, or the BIC⋆. The relevant test statistic has the form
TR/commaoriM = max
i/commaorij∈M
⏐⏐[Q(Z/commaoriˆθi) +ci]−[ Q(Z/commaoriˆθj) +cj]
⏐⏐/commaori
where cj = 2kj for the AIC, cj = log(n)kj for the BIC, cj = 2 ˆk⋆
j for the AIC⋆,
and cj = log(n) ˆk⋆
j for the BIC⋆. The computation of the resampled test statis-
tics, T∗
b/commaoriR/commaoriM/commaoriis identical for the three criteria. The reason is that the location
shift cj has no effect on the bootstrap statistics once the null hypothesis is im-
posed. Under the null hypothesis, we recenter the bootstrap statistics about
zero and this offsets the location shift ci − cj .
3.2.4. Issues Related to the Comparison of Nested Models
When two models are nested, the null hypothesis used with KLIC, E [Q(Z/commaori
θ0i)]= E[Q(Z/commaoriθ0j)]/commaorihas the strong implication that Q(Z/commaoriθ0i) = Q(Z/commaoriθ0j)
a.e. (almost everywhere), and this causes the limit distribution of the quasi-
likelihood ratio statistic,Q(Z/commaoriˆθi)−Q(Z/commaoriˆθj)/commaorito differ for nested or nonnested
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 21

THE MODEL CONFIDENCE SET 473
comparisons (see Vuong (1989)). This property of nested comparisons can be
imposed on the bootstrap resamples by replacing Q(Z/commaoriˆθj) with Q(Z∗/commaoriˆθj)/commaori
because the latter is the bootstrap variant of Q(Z/commaoriθ0j)/periodoriThe MCS pro-
cedure can be adapted so that different bootstrap schemes are used for
nested and nonnested comparisons, and imposing the stronger null hypoth-
esis Q(Z/commaoriθ0i) = Q(Z/commaoriθ0j) a.e. may improve the power of the procedure.
The key difference is that the null hypothesis with KLIC has Q(Z/commaoriˆθi) −
Q(Z/commaoriˆθj) =Op(1) for nested comparisons andQ(Z/commaoriˆθi)− Q(Z/commaoriˆθj) =Op(n1/2)
for nonnested comparisons. Our bootstrap implementation is such that
{Q(Z∗
b/commaoriˆθ∗
b/commaorii) + ˆk⋆
i − Q(Z/commaoriˆθi)}−{ Q(Z∗
b/commaoriˆθ∗
b/commaorij) + ˆk⋆
j
− Q(Z/commaoriˆθj)} is Op(n1/2),
whether the comparison involves nested or nonnested models, which causes
the bootstrap critical values to be conservative. Under the alternative,
Q(Z/commaoriˆθi) − Q(Z/commaoriˆθj) diverges at raten for nested and nonnested comparisons,
so the bootstrap testing procedure is consistent in both cases.
Since nested and nonnested comparisons result in different rates of conver-
gence and different limit distributions, there are better ways to construct an
adaptive procedure than through the test statistic TR/commaoriM, for instance, by com-
bining thep-values for the individual subhypotheses. We shall not pursue such
an adaptive bootstrap implementation in this paper. It is, however, important
to note that the issue with nested models is only relevant for KLIC because the
underlying null hypotheses of other criteria, including AIC
⋆ and BIC⋆,d on o t
imply Q(Z/commaoriθ0i) =Q(Z/commaoriθ0j) a.e. for nested models.
4. RELATION TO EXISTING MULTIPLE COMP ARISONS METHODS
The Introduction discussed the relationship between the MCS and the trace
test used to select the number of cointegration relations (seeJohansen (1988)).
The MCS and the trace test share an underlying testing principle known as
intersection–union testing (IUT). Berger (1982) was responsible for formalizing
the IUT , whilePantula (1989) applied the IUT to the problem of selecting the
lag length and order of integration in univariate autoregressive processes.
Another way to cast the MCS problem is as a multiple comparisons prob-
lem. The multiple comparisons problem has a long history in the statistics lit-
erature; see Gupta and Panchapakesan (1979), Hsu (1996), Dudoit, Shaffer,
and Boldrick (2003), and Lehmann and Romano (2005, Chap. 9) and refer-
ences therein. Results from this literature have recently been adopted in the
econometrics literature. One problem is that of multiple comparisons with best ,
where objects are compared to those with the best sample performance. Statis-
tical procedures for multiple comparisons with best are discussed and applied
to economic problems inHorrace and Schmidt(2000). Shimodaira (1998) used
a variant of Gupta’s subset selection (see Gupta and Panchapakesan (1979))
to construct a set of models that he terms a model conﬁdence set. His proce-
dure is speciﬁc to a ranking of models in terms of E(AIC
j)/commaoriand his framework
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 22

474 P . R. HANSEN, A. LUNDE, AND J. M. NASON
is different from ours in a number of ways. For instance, his preferred set of
models does not control the FWE. He also invoked a Gaussian approximation
that rules out comparisons of nested models.
Our MCS employs a sequential testing procedure that mimics step-down
procedures for multiple hypothesis testing; see, for example, Dudoit, Shaf-
fer, and Boldrick (2003), Lehmann and Romano (2005, Chap. 9), or Romano,
Shaikh, and Wolf (2008). Our deﬁnition of MCS p-values implies the mono-
tonicity, ˆpeM1
≤ ˆpeM2
≤···≤ ˆpeMm0
that is key for the result of Theorem 3.
This monotonicity is also a feature of the so-called step-down Holm adjusted
p-values.
4.1. Relationship to T ests for Superior Predictive Ability
Another related problem is the case where the benchmark, to which all ob-
jects are compared, is selected independently of the data used for the compari-
son. This problem is known as multiple comparisons with control. In the context
of forecast comparisons, this is the problem that arises when testing for supe-
rior predictive ability (SP A); seeWhite (2000b), Hansen (2005), and Romano
and Wolf (2005).
The MCS has several advantages over tests for superior predictive ability.
The reality check for data snooping of White (2000b) and the SP A test ofHansen
(2005) are designed to address whether a particular benchmark is signiﬁcantly
outperformed by any of the alternatives used in the comparison. Unlike these
tests, the MCS procedure does not require a benchmark to be speciﬁed, which
is very useful in applications without an obvious benchmark. In the situation
where there is a natural benchmark, the MCS procedure can still address the
same objective as the SP A tests. This is done by observing whether the desig-
nated benchmark is in the MCS, where the latter corresponds to a rejection of
the null hypothesis that is relevant for a SP A test.
The MCS procedure has the advantage that it can be employed for model
selection, whereas a SP A test is ill-suited for this problem. A rejection of the
SP A test only identiﬁes one or more models as signiﬁcantly better than the
benchmark.
7 Thus, the SP A test offers little guidance about which models re-
side in M∗. We are also faced with a similar problem in the event that the null
hypothesis is not rejected by the SP A test. In this case, the benchmark may be
the best model, but this label may also be applied to other models. This issue
can be resolved if all models serve as the benchmark in a series of compar-
isons. The result is a sequence of SP A tests that deﬁne the MCS to be the set
of “benchmark” models that are found not to be signiﬁcantly inferior to the
alternatives. However, the level of individual SP A tests needs to be adjusted
7Romano and Wolf (2005) improved on the reality check by identifying the entire set of alter-
natives that signiﬁcantly dominate the benchmark. This set of models is speciﬁc to the choice of
benchmark and has, therefore, no direct relation to the MCS.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 23

THE MODEL CONFIDENCE SET 475
for the number of tests that are computed to control the FWE. For example, if
the level in each of the SP A tests isα/m, the Bonferroni bound states that the
resulting set of surviving benchmarks is a MCS with coverage(1−α). Nonethe-
less, there is a substantial loss of power associated with the small level applied
to the individual tests. The loss of power highlights a major pitfall of sequential
SP A tests.
Another drawback of constructing a MCS from SP A-tests is that the null of
a SP A test is a composite hypothesis. The null is deﬁned by several inequal-
ity constraints which affect the asymptotic distribution of the SP A test statistic
because it depends on the number of binding inequalities. The binding inequal-
ity constraints create a nuisance parameter problem. This makes it difﬁcult to
control the T ype I error rate, inducing an additional loss of power; seeHansen
(2003a). In comparison, the MCS procedure is based on a sequence of hy-
pothesis tests that only involve equalities, which avoids composite hypothesis
testing.
4.2. Related Sequential T esting Procedures for Model Selection
This subsection considers some relevant aspects of out-of-sample evaluation
of forecasting models and how the MCS procedure relates to these issues.
Several papers have studied the problem of selecting the best forecasting
model from a set of competing models. For example, Engle and Brown (1985)
compared selection procedures that are based on six information criteria and
two testing procedures (general-to-speciﬁc and speciﬁc-to-general), Sin and
White (1996) analyzed information criteria for possibly misspeciﬁed models,
and Inoue and Kilian (2006) compared selection procedures that are based on
information criteria and out-of-sample evaluation. Granger, King, and White
(1995) argued that the general-to-speciﬁc selection procedure is based on an
incorrect use of hypothesis testing, because the model chosen to be the null
hypothesis in a pairwise comparison is unfairly favored. This is problematic
when the data set under investigation does not contain much information,
which makes it difﬁcult to distinguish between models. The MCS procedure
does not assume that a particular model is the true model; neither is the null
hypothesis deﬁned by a single model. Instead, all models are treated equally in
the comparison and only evaluated on out-of-sample predictive ability.
4.3. Aspects of Parameter Uncertainty and Forecasting
Parameter estimation can play an important role in the evaluation and com-
parison of forecasting models. Speciﬁcally, when the comparison of nested
models relies on parameters that are estimated using certain estimation
schemes, the limit distribution of our test statistics need not be Gaussian; see
West and McCracken (1998)a n dClark and McCracken (2001). In the present
context, there will be cases that do not fulﬁl Assumption 2. Some of these
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 24

476 P . R. HANSEN, A. LUNDE, AND J. M. NASON
problems can be avoided by using a rolling window for parameter estimation,
known as the rolling scheme . This is the approach taken by Giacomini and
White (2006). Alternatively one can estimate the parameters once (using data
that are dated prior to the evaluation period) and then compare the forecasts
conditional on these parameter estimates . However, the MCS should be applied
with caution when forecasts are based on estimated parameters because our
assumptions need not hold in this case. As a result, modiﬁcations are needed in
the case with nested models; see Chong and Hendry (1986), Harvey and New-
bold (2000), Chao, Corradi, and Swanson (2001), and Clark and McCracken
(2001) among others. The key modiﬁcation that is needed to accommodate
the case with nested models is to adopt a test with a proper size. With proper
choices for δ
M and eM, the general theory for the MCS procedure remains.
However, in this paper we will not pursue this extension because it would ob-
scure our main objective, which is to lay out the key ideas of the MCS.
4.4. Bayesian Interpretation
The MCS procedure is based on frequentist principles, but resembles some
aspects of Bayesian model selection techniques. By specifying a prior over the
models in M0, a Bayesian procedure would produce a posterior distribution
for each model, conditional on the actual data. This approach to MCS con-
struction includes those models with the largest posteriors that sum at least to
1 − α/periodoriIf the Bayesian were also to choose models by minimizing the “risk” as-
sociated with the loss attributed to each model, the MCS would be a Bayes de-
cision procedure with respect to the model posteriors. Note that the Bayesian
and frequentist MCSs rely on the metric under which loss is calculated and
depend on sample information.
We argue that our approach to the MCS and its bootstrap implementation
compares favorably to Bayesian methods of model selection. One advantage
of the frequentist approach is that it avoids having to place priors on the ele-
ments of
M0 (and their parameters). Our probability statement is associated
with the random data-dependent set of models that is the MCS. It therefore is
meaningful to state that the best model can be found in the MCS with a cer-
tain probability. The MCS also places moderate computational demands on
the researcher, unlike the synthetic data creation methods on which Bayesian
Markov chain Monte Carlo methods rely.
5. SIMULATION RESULTS
This section reports on Monte Carlo experiments that show the MCS to be
properly sized and possess good power in various simulation designs.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 25

THE MODEL CONFIDENCE SET 477
5.1. Simulation Experiment I
We consider two designs that are based on the m-dimensional vector
θ= (0/commaori1
m−1/commaori/periodori/periodori/periodori/commaorim−2
m−1/commaori1)′λ/√n that deﬁnes the relative performances μij =
E(dij/commaorit) =θi − θj . The experimental design ensures that M∗ consists of a single
element, unlessλ= 0, in which case we have M∗ = M0. The stochastic nature
of the simulation is primarily driven by
Xt ∼ i/periodorii/periodorid/periodoriNm(0/commaoriΣ)/commaoriwhere
Σij =
{ 1f o r i =j,
ρ for i ̸=j, for some 0 ≤ ρ≤ 1,
where ρcontrols the degree of correlation between alternatives.
DESIGN I.A—Symmetric Distributed Loss: Deﬁne the (vector of) loss vari-
ables to be
Lt ≡ θ+ at
√
E(a2
t )
Xt/commaoriwhere
at = exp(yt)/commaori yt = −ϕ
2(1 +ϕ)+ϕyt−1 + √ϕεt/commaori
and εt ∼ i/periodorii/periodorid/periodoriN(0/commaori1)/periodoriThis implies that E(yt) =− ϕ/{2(1 − ϕ2)} and var(yt) =
ϕ/(1−ϕ2) such that E(at) = exp{E(yt)+var(yt)/2}= exp{0}= 1a n dv a r(at) =
(exp{ϕ/(1 − ϕ2)}− 1). Furthermore, E(a2
t ) = var(at) + 1 = exp{ϕ/(1 − ϕ2)}
such that var (Lt) = 1. Note that ϕ= 0 corresponds to homoskedastic er-
rors and ϕ> 0 corresponds to (generalized autoregressive conditional het-
eroskedasticity) (GARCH type) heteroskedastic errors.
The simulations employ 2,500 repetitions, where λ= 0, 5, 10, 20, ρ= 0/periodori00,
0.50, 0.75, 0.95, ϕ= 0/periodori0, 0.5, 0.8, and m = 10, 40, 100. We use the block boot-
strap, in which blocks have length l = 2, and results are based on B = 1/commaori000
resamples. The size of a synthetic sample is n = 250. This approximates sam-
ple sizes often available for model selection exercises in macroeconomics.
We report two statistics from our simulation experiment based onα= 10%:
one is the frequency at which ˆM∗
90% contains M∗; the other is the average
number of models in ˆM∗
90%
. The former shows the size properties of the MCS
procedure; the latter is informative about the power of the procedure.
Ta b l eII presents simulation results that show that the small sample prop-
erties of the MCS procedure closely match its theoretical predictions. The
frequency that the best models are contained in the MCS is almost always
greater than (1 − α), and the MCS becomes better at separating the infe-
rior models from the superior model, as the μijs become more disperse (e.g.,
as λincreases). Note also that a larger correlation makes it easier to sep-
arate inferior models from superior model. This is not surprising because
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 26

478 P . R. HANSEN, A. LUNDE, AND J. M. NASON
TABLE II
SIMULATION DESIGN I.Aa
m = 10 m = 40 m = 100
λρ = 00 /periodori50 /periodori75 0 /periodori95 0 0 /periodori50 /periodori75 0 /periodori95 0 0 /periodori50 /periodori75 0 /periodori95
Panel A:ϕ= 0
Frequency at which M∗ ⊂ ˆM∗
90% (size)
0 0.885 0.898 0.884 0.885 0 /periodori882 0 /periodori882 0 /periodori877 0 /periodori880 0 /periodori880 0 /periodori870 0 /periodori877 0 /periodori875
5 0.990 0.988 0.991 1.000 0 /periodori980 0 /periodori979 0 /periodori976 0 /periodori984 0 /periodori975 0 /periodori976 0 /periodori975 0 /periodori976
10 0.994 0.998 0.999 1.000 0 /periodori978 0 /periodori983 0 /periodori985 0 /periodori993 0 /periodori973 0 /periodori975 0 /periodori974 0 /periodori980
20 0.998 1.000 1.000 1.000 0 /periodori988 0 /periodori981 0 /periodori991 1 /periodori000 0 /periodori975 0 /periodori978 0 /periodori986 0 /periodori992
40 1.000 1.000 1.000 1.000 0 /periodori992 0 /periodori996 0 /periodori998 1 /periodori000 0 /periodori981 0 /periodori984 0 /periodori990 0 /periodori998
Average number of elements in ˆM∗
90% (power)
0 9.614 9.658 9.646 9.632 38 /periodori68 38 /periodori78 38 /periodori91 38 /periodori82 97 /periodori02 96 /periodori84 97 /periodori11 97 /periodori20
5 6.498 4.693 3.239 1.544 25 /periodori30 18 /periodori79 13 /periodori35 6 /periodori382 59 /periodori87 43 /periodori92 32 /periodori51 15 /periodori04
10 3.346 2.390 1.732 1.027 13 /periodori59 9 /periodori829 7 /periodori142 3 /periodori266 32 /periodori32 23 /periodori04 16 /periodori97 7 /periodori902
20 1.702 1.307 1.062 1.000 7 /periodori060 5 /periodori010 3 /periodori617 1 /periodori674 17 /periodori03 12 /periodori40 8 /periodori785 4 /periodori049
40 1.072 1.005 1.000 1.000 3 /periodori572 2 /periodori597 1 /periodori840 1 /periodori052 8 /periodori778 6 /periodori375 4 /periodori521 2 /periodori083
Panel B:ϕ= 0/periodori5
Frequency at which M∗ ⊂ ˆM∗
90% (size)
0 0.908 0.897 0.905 0.894 0 /periodori911 0 /periodori907 0 /periodori910 0 /periodori916 0 /periodori925 0 /periodori918 0 /periodori909 0 /periodori913
5 0.985 0.990 0.995 1.000 0 /periodori971 0 /periodori976 0 /periodori977 0 /periodori987 0 /periodori974 0 /periodori974 0 /periodori973 0 /periodori973
10 0.992 0.999 1.000 1.000 0 /periodori978 0 /periodori985 0 /periodori982 0 /periodori995 0 /periodori975 0 /periodori969 0 /periodori983 0 /periodori984
20 0.999 1.000 1.000 1.000 0 /periodori988 0 /periodori989 0 /periodori988 1 /periodori000 0 /periodori979 0 /periodori976 0 /periodori981 0 /periodori992
40 1.000 1.000 1.000 1.000 0 /periodori996 0 /periodori996 1 /periodori000 1 /periodori000 0 /periodori980 0 /periodori982 0 /periodori991 0 /periodori999
Average number of elements in ˆM∗
90%
(power)
0 9.660 9.664 9.664 9.649 38 /periodori97 38 /periodori93 39 /periodori03 39 /periodori05 98 /periodori35 98 /periodori05 97 /periodori94 97 /periodori73
5 6.076 4.497 3.213 1.564 24 /periodori33 17 /periodori72 13 /periodori13 6 /periodori112 57 /periodori84 41 /periodori60 30 /periodori35 14 /periodori54
10 3.188 2.278 1.680 1.035 12 /periodori95 9 /periodori268 6 /periodori791 3 /periodori136 30 /periodori54 22 /periodori30 16 /periodori56 7 /periodori510
20 1.700 1.274 1.069 1.000 6 /periodori819 4 /periodori883 3 /periodori563 1 /periodori659 16 /periodori04 11 /periodori56 8 /periodori430 3 /periodori894
40 1.085 1.008 1.000 1.000 3 /periodori506 2 /periodori517 1 /periodori811 1 /periodori061 8 /periodori339 6 /periodori166 4 /periodori360 2 /periodori034
Panel C:ϕ= 0/periodori8
Frequency at which M∗ ⊂ ˆM∗
90% (size)
0 0.931 0.940 0.939 0.947 0 /periodori963 0 /periodori968 0 /periodori958 0 /periodori962 0 /periodori970 0 /periodori975 0 /periodori969 0 /periodori972
5 0.990 0.997 0.998 1.000 0 /periodori977 0 /periodori980 0 /periodori989 0 /periodori993 0 /periodori970 0 /periodori975 0 /periodori976 0 /periodori981
10 0.998 1.000 1.000 1.000 0 /periodori984 0 /periodori987 0 /periodori992 0 /periodori998 0 /periodori982 0 /periodori976 0 /periodori974 0 /periodori991
20 1.000 1.000 1.000 1.000 0 /periodori990 0 /periodori993 0 /periodori996 1 /periodori000 0 /periodori982 0 /periodori982 0 /periodori992 0 /periodori998
40 1.000 1.000 1.000 1.000 0 /periodori999 1 /periodori000 1 /periodori000 1 /periodori000 0 /periodori988 0 /periodori994 0 /periodori996 1 /periodori000
Average number of elements in ˆM∗
90% (power)
0 9.739 9.814 9.794 9.799 39 /periodori61 39 /periodori61 39 /periodori53 39 /periodori55 99 /periodori00 99 /periodori44 99 /periodori15 99 /periodori43
5 4.301 3.318 2.386 1.322 16 /periodori26 12 /periodori31 9 /periodori118 4 /periodori401 39 /periodori69 28 /periodori13 20 /periodori56 10 /periodori12
10 2.424 1.864 1.419 1.062 9 /periodori133 6 /periodori643 4 /periodori727 2 /periodori349 20 /periodori72 14 /periodori77 11 /periodori26 5 /periodori470
20 1.455 1.220 1.092 1.010 4 /periodori770 3 /periodori520 2 /periodori535 1 /periodori454 11 /periodori15 8 /periodori014 5 /periodori948 2 /periodori840
40 1.098 1.037 1.011 1.003 2 /periodori645 1 /periodori967 1 /periodori490 1 /periodori081 5 /periodori932 4 /periodori356 3 /periodori248 1 /periodori645
aThe two statistics are the frequency at which ˆM∗
90% contains M∗ and the other is the average number of models
in ˆM∗
90%. The former shows the ‘size’ properties of the MCS procedure and the latter is informative about the ‘power’
of the procedure.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 27

THE MODEL CONFIDENCE SET 479
var(dij/commaorit) = var(Lit) + var(Ljt) − 2c o v(Lit/commaoriLjt) = 2(1 − ρ)/commaoriwhich is decreas-
ing in ρ. Thus, a larger correlation (holding the individual variances ﬁxed) is
associated with more information that allows the MCS to separate good from
bad models. Finally, the effects of heteroskedasticity are relatively small, but
heteroskedasticity does appear to add power to the MCS procedure. The av-
e r a g en u m b e ro fm o d e l si nˆM∗
90% tends to fall as ϕincreases.
Corollary 1 has a consistency result that applies when λ>0. The implica-
tion is that only one model enters M∗ under this restriction. T able II shows
that M∗ often contains only one model given λ>0. The MCS matches this
theoretical prediction in T able II because ˆM∗
90%
= M∗ in a large number of
simulations. This equality holds especially when λand ρare large. These are
also the simulation experiments that yield size and power statistics equal (or
nearly equal) to 1. With size close to 1 or equal to 1, observe that M∗ ⊂ ˆM∗
90%
(in all the synthetic samples). On the other hand, ˆM∗
90%
is reduced to a single
model (in all the synthetic samples) when power is close to 1 or equal to 1.
DESIGN I.B—Dependent Loss: This design sets Lt ∼ i/periodorii/periodorid/periodoriN10(θ/commaoriΣ),w h e r e
the covariance matrix has the structureΣij =ρ|i−j| for ρ= 0/commaori0/periodori5, and 0/periodori75. The
mean vector takes the formθ=(0/commaori/periodori/periodori/periodori/commaori0/commaori1
5/commaori/periodori/periodori/periodori/commaori1
5)′ so that the number of zero
elements in θdeﬁnes the number of elements in M∗/periodoriWe report simulation
results for the case where m0 = 10 and M∗ consists of either one, two, or ﬁve
models.
The simulation results are presented in Figure 1. The left panels display the
frequency at which ˆM∗
90% contains M∗ (size) at various sample sizes. The right
panels present the average number of models in ˆM∗
90%
(power). The two upper
panels contain the results for the case where M∗ is a single model. The upper-
left panel indicates that the best model is almost always contained in the MCS.
This agrees with Corollary 1, which states that ˆM∗
1−α
p
→ M∗ as n →∞ /commaoriwhen-
ever M∗ consists of a single model. The upper-right panel illustrates the power
of the procedure based onTmax/commaoriM = maxi∈M ti·.W en o t et h a ti tt a k e sa b o u t8 0 0
observations to weed out the 9 inferior models in this design. The MCS pro-
cedure is barely affected by the correlation parameter ρ/commaorib u tw en o t et h a ta
largerρresults in a small loss in power. In the lower-left panel, we see that the
frequency at which M∗ is contained in ˆM∗
90% is reasonably close to 90% except
for the very short sample sizes. From the middle-right and lower-right panels,
we see that it takes about 500 observations to remove all the poor models.
The middle-right and lower-right panels illustrate another aspect of the MCS
procedure. For large sample sizes, we note that the average number of models
in ˆM∗
90% falls below the number of models in M∗/periodoriThe explanation is sim-
ple. After all poor models have been eliminated, as occurs with probability
approaching 1 as n →∞ /commaorithere is a positive probability that H0/commaoriM∗ is rejected,
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 28

480 P . R. HANSEN, A. LUNDE, AND J. M. NASON
FIGURE 1.—Simulation Design I.B with 10 alternatives and 1, 2, or 5 elements in M∗.T h el e f t
panels report the frequency at which M∗ is contained in ˆM∗
90% (size properties) and the right
panels report the average number of models in ˆM∗
90% (power properties).
which causes the MCS procedure to eliminate a good model. Thus, the infer-
ences we draw from the simulation results are quite encouraging for theTmax/commaoriM
test.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 29

THE MODEL CONFIDENCE SET 481
5.2. Simulation Experiment II: Regression Models
Next we study the properties of the MCS procedure in the context of in-
sample evaluation of regression models as we laid out in Section 3.2.W ec o n -
sider a setup with six potential regressors, Xt =(X1/commaorit/commaori/periodori/periodori/periodori/commaoriX6/commaorit)′/commaorithat are dis-
tributed as
Xt ∼ i/periodorii/periodorid/periodoriN6(0/commaoriΣ)/commaoriwhere
Σij =
{ 1f o r i =j/commaori
ρ for i ̸=j/commaorifor some 0 ≤ ρ<1,
where ρmeasures the degree of dependence between the regressors. We
deﬁne the dependent variable by Yt = μ+ βX1/commaorit+
√
1 − β2εt ,w h e r eεt ∼
i/periodorii/periodorid/periodoriN(0/commaori1). In addition to the six variables in Xt/commaoriwe include a constant,
X0/commaorit= 1/commaoriin all regression models. The set of regressions being estimated is
given by the 12 regression models that are listed in each of the panels in T a-
ble III.
We report simulation results based on 10,000 repetitions, using a design with
an R2 = 50% (i.e., β2 = 0/periodori5) and either ρ= 0/periodori3o r ρ= 0/periodori9.8 For the number of
bootstrap resamples, we use B = 1,000. Since X0/commaorit= 1 is included in all regres-
sion models, the relevant MCS statistics are invariant to the actual value forμ,
so we set μ= 0 in our simulations.
The deﬁnition of M∗ will depend on the criterion. With KLIC, the set of best
models is given by the set of regression models that includes X1/periodoriThe reason
is that KLIC does not favor parsimonious models, unlike the AIC ⋆ and BIC⋆.
With these two criteria, M∗ is deﬁned to be the most parsimonious regression
model that includesX1.T h em o d e l si nM∗ are identiﬁed by the shaded regions
in T ableIII.
Our simulation results are reported in T able III. The average value of
Q(Zj/commaoriˆθj) is given in the ﬁrst pair of data columns, followed by the average
estimate of the effective degrees of freedom, ˆk⋆/periodoriThe Gaussian setup is such
that all models are correctly speciﬁed, so the effective degrees of freedom is
simply the number of free parameters, which is the number of regressors plus
1f o rσ
2
j /periodoriTa b l eIII shows that the average value ofˆk⋆
j is very close to the number
of free parameters in thejth regression model. The last three pairs of columns
report the frequency that each of the models are in ˆM∗
90%
/periodoriWe want large num-
bers inside the shaded region and small numbers outside the shaded region.
The results are intuitive. As the sample size increases from 50 to 100 and then
to 500, the MCS procedure becomes better at eliminating the models that do
not reside in
M∗/periodoriWith a sample size ofn = 500/commaorithe consistent criterion, BIC⋆,
8Simulation results for β2 = 0/periodori1a n d0 /periodori9 are available in a separate appendix; see Hansen,
Lunde, and Nason (2011).
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 30

482 P . R. HANSEN, A. LUNDE, AND J. M. NASON
TABLE III
SIMULATION EXPERIMENT IIa
Q(Zj/commaoriˆθj) ˆk⋆ KLIC AIC ⋆ (TIC) BIC ⋆
ρ= 0/periodori30 /periodori90 /periodori30 /periodori90 /periodori30 /periodori90 /periodori30 /periodori90 /periodori30 /periodori9
Panel A:n = 50
X0 48/periodori14 8 /periodori1 1.99 2.00 0.058 0.038 0.085 0.070 0.118 0.124
X0/commaoriX1 12/periodori41 2 /periodori4 3.02 3.02 0.998 0.999 1.000 1.000 1.000 1.000
X0/commaori/periodori/periodori/periodori/commaoriX2 11/periodori31 1 /periodori3 4.08 4.08 0.998 0.999 0.962 0.999 0.566 0.940
X0/commaori/periodori/periodori/periodori/commaoriX3 10/periodori21 0 /periodori2 5.18 5.18 0.999 0.999 0.940 0.998 0.469 0.912
X0/commaori/periodori/periodori/periodori/commaoriX4 9/periodori09 9 /periodori04 6.32 6.32 1.000 1.000 0.905 0.997 0.367 0.803
X0/commaori/periodori/periodori/periodori/commaoriX5 7/periodori95 7 /periodori88 7.50 7.50 1.000 1.000 0.867 0.994 0.279 0.598
X0/commaori/periodori/periodori/periodori/commaoriX6 6/periodori77 6 /periodori69 8.73 8.74 1.000 1.000 0.806 0.990 0.203 0.400
X0/commaoriX2 44/periodori72 1 /periodori0 3.02 3.02 0.086 0.905 0.100 0.935 0.099 0.877
X0/commaoriX2/commaoriX3 42/periodori31 8 /periodori1 4.08 4.08 0.106 0.948 0.107 0.949 0.077 0.806
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX4 40/periodori41 6 /periodori3 5.18 5.18 0.120 0.958 0.105 0.938 0.054 0.665
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX5 38/periodori81 4 /periodori8 6.32 6.32 0.132 0.962 0.100 0.913 0.036 0.501
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX6 37/periodori21 3 /periodori4 7.50 7.51 0.145 0.964 0.094 0.869 0.022 0.348
Panel B:n = 100
X0 98/periodori09 8 /periodori1 1.99 1.99 0.000 0.000 0.000 0.000 0.000 0.000
X0/commaoriX1 27/periodori62 7 /periodori8 3.00 3.00 0.998 1.000 1.000 1.000 1.000 1.000
X0/commaori/periodori/periodori/periodori/commaoriX2 26/periodori62 6 /periodori7 4.03 4.03 0.999 1.000 0.959 0.982 0.402 0.675
X0/commaori/periodori/periodori/periodori/commaoriX3 25/periodori52 5 /periodori7 5.07 5.06 0.999 1.000 0.939 0.975 0.276 0.619
X0/commaori/periodori/periodori/periodori/commaoriX4 24/periodori42 4 /periodori6 6.12 6.12 1.000 1.000 0.908 0.960 0.174 0.545
X0/commaori/periodori/periodori/periodori/commaoriX5 23/periodori42 3 /periodori6 7.19 7.18 1.000 1.000 0.864 0.942 0.101 0.390
X0/commaori/periodori/periodori/periodori/commaoriX6 22/periodori32 2 /periodori5 8.28 8.27 1.000 1.000 0.800 0.920 0.059 0.238
X0/commaoriX2 92/periodori44 5 /periodori1 3.00 3.01 0.000 0.548 0.000 0.585 0.000 0.490
X0/commaoriX2/commaoriX3 88/periodori84 0 /periodori4 4.03 4.03 0.000 0.691 0.000 0.666 0.000 0.443
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX4 86/periodori13 8 /periodori1 5.07 5.07 0.000 0.736 0.000 0.675 0.000 0.338
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX5 83/periodori93 6 /periodori3 6.12 6.12 0.000 0.759 0.000 0.655 0.000 0.236
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX6 82/periodori03 4 /periodori8 7.19 7.19 0.001 0.772 0.000 0.631 0.000 0.143
Panel C:n = 500
X0 498 498 2.00 2.00 0.000 0.000 0.000 0.000 0.000 0.000
X0/commaoriX1 151 151 3.00 3.00 0.999 0.999 1.000 1.000 1.000 1.000
X0/commaori/periodori/periodori/periodori/commaoriX2 150 150 4.00 4.00 0.999 0.999 0.958 0.960 0.207 0.206
X0/commaori/periodori/periodori/periodori/commaoriX3 149 149 5.01 5.01 0.999 1.000 0.938 0.938 0.100 0.099
X0/commaori/periodori/periodori/periodori/commaoriX4 148 148 6.02 6.01 1.000 1.000 0.907 0.901 0.044 0.042
X0/commaori/periodori/periodori/periodori/commaoriX5 147 147 7.03 7.02 1.000 1.000 0.858 0.852 0.020 0.017
X0/commaori/periodori/periodori/periodori/commaoriX6 145 146 8.04 8.03 1.000 1.000 0.790 0.792 0.006 0.008
X0/commaoriX2 474 238 3.00 3.00 0.000 0.000 0.000 0.000 0.000 0.000
X0/commaoriX2/commaoriX3 460 219 4.00 4.00 0.000 0.002 0.000 0.002 0.000 0.002
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX4 451 211 5.01 5.01 0.000 0.004 0.000 0.004 0.000 0.001
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX5 444 206 6.02 6.01 0.000 0.006 0.000 0.006 0.000 0.001
X0/commaoriX2/commaori/periodori/periodori/periodori/commaoriX6 439 203 7.03 7.02 0.000 0.008 0.000 0.007 0.000 0.000
aThe average value of the maximized log-likelihood function multiplied by − 2 is reported in the ﬁrst two data
columns. The next pair of columns has the average of the effective degrees of freedom. The last three pairs of columns
report the frequency that a particular regression model is in the ˆM∗
90% for each of the three criteria: KLIC, AIC ⋆,
and BIC⋆.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 31

THE MODEL CONFIDENCE SET 483
has reduced the MCS to the single best model in the majority of simulations /periodori
This is not true for the AIC⋆ criterion. Although it tends to settle on more par-
simonious models than the KLIC, the AIC⋆ has a penalty that makes it possible
for an overparameterized model to have the best AIC⋆/periodoriThe bootstrap testing
procedure is conservative when the comparisons involve nested models under
KLIC; see our discussion in the last paragraph of Section 3.2. This explains
that both T ype I and T ype II errors are close to zero when n = 500/commaorian ideal
outcome that is not guaranteed when M∗
KLIC includes nonnested models.9
6. EMPIRICAL APPLICATIONS
6.1. U.S. Inﬂation Forecasts: Stock and Watson (1999) Revisited
This section revisits the Stock and Watson (1999) study of the best out-of-
sample predictors of inﬂation. Their empirical application consists of pairwise
comparisons of a large number of inﬂation forecasting models. The set of inﬂa-
tion forecasting models includes several that have a Phillips curve interpreta-
tion, along with autoregressive and a no-change (month-over-month) forecast.
We extend their set of forecasts by adding a second no-change (12-months-
over-12-months) forecast that was used in Atkeson and Ohanian (2001).
Stock and Watson (1999) measured inﬂation, π
t , as either the CPI-U, all
items (PUNEW), or the headline personal consumption expenditure implicit
price deﬂator (GMDC).10 The relevant Phillips curve is
πt+h − πt =φ+β(L)ut +γ(L)(1 − L)πt +et+h/commaori(4)
where ut is the unemployment rate, L is the lag polynomial operator, and et+h
is the long-horizon inﬂation forecast innovation. Note that the natural rate
hypothesis is not imposed on the Phillips curve ( 4) and that inﬂation as a re-
gressor is in its ﬁrst difference. Stock and Watson also forecasted inﬂation with
(4) where the unemployment rateut is replaced with different macrovariables.
The entire sample runs from 1959:M1 to 1997:M9. Following Stock and Wat-
son, we study the properties of their forecasting models on the pre- and post-
1984 subsamples of 1970:
M1–1983:M12 and 1984: M1–1996:M9.11 The former
subsample contains the great inﬂation of the 1970s and the rapid disinﬂation
of the early 1980s. Inﬂation does not exhibit this volatile behavior in the post-
1984 subsample. We follow Stock and Watson so as to replicate their inﬂation
9In an unreported simulation study where M∗
KLIC was designed to include nonnested models,
we found the frequency by which M∗
KLIC ⊂ ˆM∗
90%
converges to 90%/periodori
10The data for this applications was downloaded from Mark Watson’s web page. We refer the
interested reader to Stock and Watson(1999) for details about the data and model speciﬁcations.
11Stock and Watson split their sample at the end of 1983 to account for structural change in
inﬂation dynamics. This structural break is ignored when estimating the Phillips curve model (4)
and the alternative inﬂation forecasting equations. This is justiﬁed by Stock and Watson because
the impact of the 1984 structural break on their estimated Phillips curve coefﬁcients is small.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 32

484 P . R. HANSEN, A. LUNDE, AND J. M. NASON
forecasts. However, our MCS bootstrap implementation, which is described
in Section 3, relies on an assumption that dij/commaoritis stationary. This is not plau-
sible when the parameters are estimated with a recursive estimation scheme,
as was used by Stock and Watson (1999). We avoid this problem by following
Giacomini and White (2006) and present empirical results that are based on
parameters estimated over a rolling window with a ﬁxed number of observa-
tions.
12 Regressions are estimated on data that begin no earlier than 1960: M2,
although lagged regressors impinge on observations back to 1959:M1.
We compute the MCS across all of the Stock and Watson inﬂation forecast-
ing models. This includes the Phillips curve model (4), the inﬂation forecasting
equation that runs through all of the macrovariables considered by Stock and
Watson, a univariate autoregressive model, and two no-change forecasts. The
ﬁrst no-change forecast is the past month’s inﬂation rate; the second no-change
forecast uses the past year’s inﬂation rate as its forecast. The former matches
the no-change forecast in Stock and Watson (1999) and the latter matches the
no-change forecast in Atkeson and Ohanian (2001). Stock and Watson also
presented results for forecast combinations and forecasts based on principal
component indicator variables.
13
Ta b l e sIV and V report (the level of) the root mean square error (RMSE)
and MCSp-values for each of the inﬂation forecasting models. The second col-
umn of T ableIV also lists the transformation of the macrovariable employed
by the forecasting equation.
Our T ableIV matches the results reported in Stock and Watson (1999,T a -
ble 2). The initial model spaceM0 is ﬁlled with a total of 19 models. The results
for the two no-change forecasts and the AR(p) are the ﬁrst three rows of T a-
ble IV. The RMSEs and the p-values for the Phillips curve forecasting model
(4) appear in the bottom row of our T able IV. The rest of the rows of T a-
ble IV are the “gap” and “ﬁrst difference” speciﬁcations of Stock and Watson’s
aggregate activity variables that appear in place of ut in inﬂation forecasting
equation (4). The gap variables are computed with a one-sided Hodrick and
Prescott (1997) ﬁlter; see Stock and Watson (1999, p. 301) for details.14
A glance at T able IV reveals that the MCS of subsamples 1970: M1–
1983:M12 and 1984: M1–1996:M9 are strikingly different for both inﬂation se-
ries, PUNEW and GMDC. The MCS of the pre-1984 subsample places seven
12The corresponding empirical results that are based on parameters that are estimated with the
recursive scheme, as was used in Stock and Watson (1999), are available in a separate appendix;
see Hansen, Lunde, and Nason (2011). Although our assumption does not justify the recursive
estimation scheme, it produces pseudo-MCS results that are very similar to those obtained under
the rolling window estimation scheme.
13See Stock and Watson (1999) for details about their modelling strategy, forecasting proce-
dures, and data set.
14The MCS p-values are computed using a block size of l = 12 in the bootstrap implementa-
tion. The MCS p-values are qualitatively similar when computed with l = 6a n dl = 9/periodoriThese are
reported in a separate appendix; see Hansen, Lunde, and Nason (2011).
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 33

THE MODEL CONFIDENCE SET 485
TABLE IV
MCS FOR SIMPLE REGRESSION-BASED INFLATION FORECASTSa
PUNEW GMDC
1970–1983 1984–1996 1970–1983 1984–1996
V ariable T rans RMSE pMCS RMSE pMCS RMSE pMCS RMSE pMCS
No change (month) 3.290 0.001 2.140 0.122 ∗ 2.208 0.042 1.751 0.113 ∗
No change (year) – 2.798 0.006 1.207 1.00 ∗∗ 2.100 0.109 ∗ 0.888 1.00 ∗∗
uniar – 2.802 0.004 1.330 0.736 ∗∗ 2.026 0.145 ∗ 1.070 0.411 ∗∗
Gap speciﬁcations
dtip DT 2.597 0.059 1.475 0.651 ∗∗ 2.103 0.095 1.050 0.411 ∗∗
dtgmpyq DT 2.751 0.020 1.691 0.299 ∗∗ 2.090 0.157 ∗ 1.125 0.317 ∗∗
dtmsmtq DT 2.202 0.872 ∗∗ 1.704 0.477 ∗∗ 1.806 0.464 ∗∗ 1.046 0.411 ∗∗
dtlpnag DT 2.591 0.068 1.433 0.694 ∗∗ 2.132 0.075 1.026 0.411 ∗∗
ipxmca LV 2.609 0.034 1.318 0.736 ∗∗ 2.040 0.261 ∗∗ 1.034 0.411 ∗∗
hsbp LN 2.114 1.00 ∗∗ 1.582 0.579 ∗∗ 1.967 0.364 ∗∗ 1.034 0.411 ∗∗
lhmu25 LV 2.968 0.006 1.439 0.651 ∗∗ 2.231 0.061 1.040 0.411 ∗∗
First difference speciﬁcations
ip DLN 2.344 0.306
∗∗ 1.393 0.736 ∗∗ 1.946 0.298 ∗∗ 1.058 0.411 ∗∗
gmpyq DLN 2.306 0.842 ∗∗ 1.524 0.421 ∗∗ 1.709 1.00 ∗∗ 1.158 0.317 ∗∗
msmtq DLN 2.158 0.872 ∗∗ 1.391 0.736 ∗∗ 1.857 0.464 ∗∗ 1.066 0.411 ∗∗
lpnag DLN 2.408 0.430 ∗∗ 1.341 0.736 ∗∗ 1.940 0.298 ∗∗ 1.027 0.411 ∗∗
dipxmca DLV 2.379 0.139 ∗ 1.353 0.736 ∗∗ 1.903 0.446 ∗∗ 1.041 0.411 ∗∗
dhsbp DLN 2.850 0.003 1.456 0.665 ∗∗ 2.076 0.075 1.070 0.411 ∗∗
dlhmu25 DLV 2.383 0.169 ∗ 1.440 0.579 ∗∗ 2.035 0.102 ∗ 1.065 0.411 ∗∗
dlhur DLV 2.296 0.631 ∗∗ 1.429 0.691 ∗∗ 1.904 0.330 ∗∗ 1.067 0.411 ∗∗
Phillips curve
lhur 2.637 0.034 1.388 0.736 ∗∗ 2.076 0.098 1.162 0.325 ∗∗
aRMSEs and MCS p-values for the different forecasts. The forecasts in ˆM∗
90% and ˆM∗
75%
are identiﬁed by one
and two asterisks, respectively.
forecasting models in PUNEW- ˆM∗
75% and nine models in GMDC- ˆM∗
75%
.F o r
the post-1984 subsample, all but one model ends up inˆM∗
75%
for both PUNEW
and GMDC. The only model that is consistently kicked out of these MCSs is
the monthly no-change forecast, which uses last month’s inﬂation rate as its
forecast.
Another intriguing feature of T able IV is the inﬂation forecasting models
that reside in the MCS when faced with the 1970:
M1–1983:M12 subsample.
The seven models that are in PUNEW- ˆM∗
75% are driven by macrovariables
related either to real economic activity (e.g., manufacturing and trade, and
building permits) or to the labor market. The labor market variables are lp-
nag (employees on nonagricultural payrolls) and dlhur (ﬁrst difference of the
unemployment rate, all workers 16 years and older). Thus, there is labor mar-
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 34

486 P . R. HANSEN, A. LUNDE, AND J. M. NASON
TABLE V
MCS RESULTS FOR SHRINKAGE-TYPE INFLATION FORECASTSa
PUNEW GMDC
1970–1983 1984–1996 1970–1983 1984–1996
V ariable RMSE pMCS RMSE pMCS RMSE pMCS RMSE pMCS
No change (month) 3.290 0.006 2.140 0.000 2.208 0.006 1.751 0.000
No change (year) 2.798 0.020 1.207 1.00 ∗∗ 2.100 0.120 ∗ 0.888 1.00 ∗∗
Univariate 2.802 0.012 1.330 0.718 ∗∗ 2.026 0.046 1.070 0.378 ∗∗
Panel A. All indicators
Mul. factors 2.367 0.266 ∗∗ 1.407 0.069 2.105 0.088 1.013 0.570 ∗∗
1 factor 2.106 1.00 ∗∗ 1.351 0.186 ∗ 1.746 1.00 ∗∗ 1.038 0.570 ∗∗
Comb. mean 2.423 0.093 1.269 0.869 ∗∗ 1.880 0.585 ∗∗ 1.030 0.570 ∗∗
Comb. median 2.585 0.030 1.294 0.869 ∗∗ 1.939 0.323 ∗∗ 1.055 0.530 ∗∗
Comb. ridge reg. 2.121 0.975 ∗∗ 1.318 0.869 ∗∗ 1.918 0.518 ∗∗ 1.013 0.570 ∗∗
Panel B. Real activity indicators
Mul. factors 2.245 0.768 ∗∗ 1.416 0.022 1.959 0.323 ∗∗ 0.990 0.570 ∗∗
1 factor 2.115 0.975 ∗∗ 1.347 0.358 ∗∗ 1.774 0.720 ∗∗ 1.041 0.570 ∗∗
Comb. mean 2.284 0.615 ∗∗ 1.263 0.869 ∗∗ 1.827 0.698 ∗∗ 1.012 0.570 ∗∗
Comb. median 2.329 0.495 ∗∗ 1.284 0.869 ∗∗ 1.854 0.647 ∗∗ 1.038 0.553 ∗∗
Comb. ridge reg. 2.160 0.953 ∗∗ 1.326 0.855 ∗∗ 1.888 0.518 ∗∗ 1.013 0.570 ∗∗
Panel C. Interest rates
Mul. factors 2.828 0.019 1.512 0.005 2.215 0.008 1.294 0.008
1 factor 2.776 0.030 1.463 0.003 2.111 0.007 1.102 0.161 ∗
Comb. mean 2.474 0.092 1.349 0.123 ∗ 1.935 0.323 ∗∗ 1.060 0.522 ∗∗
Comb. median 2.567 0.077 1.377 0.034 1.974 0.290 ∗∗ 1.066 0.418 ∗∗
Comb. ridge reg. 2.436 0.164 ∗ 1.372 0.069 1.962 0.216 ∗ 1.052 0.530 ∗∗
Panel D. Money
Mul. factors 2.801 0.015 1.340 0.597 ∗∗ 2.028 0.020 1.075 0.057
1 factor 2.805 0.013 1.352 0.186 ∗ 2.027 0.031 1.104 0.026
Comb. mean 2.742 0.019 1.390 0.022 2.033 0.012 1.088 0.015
Comb. median 2.752 0.019 1.340 0.386 ∗∗ 2.032 0.008 1.077 0.095
Comb. ridge reg. 2.721 0.019 1.446 0.007 2.013 0.088 1.088 0.010
Phillips curve
LHUR 2.637 0.030 1.388 0.022 2.076 0.031 1.162 0.423 ∗∗
aRMSEs and MCS p-values for the different forecasts. The forecasts in ˆM∗
90% and ˆM∗
75%
are identiﬁed by one
and two asterisks, respectively.
ket information that is important for predicting inﬂation during the pre-1984
subsample. This result is consistent with traditional Keynesian measures of ag-
gregate demand.
Ta b l eIV also shows that there are two levels and ﬁve ﬁrst difference speciﬁ-
cations of the forecasting equation that consistently appear in ˆM∗
75% using the
1970:M1–1983:M12 subsample. On this subsample, only msmtq (total real man-
ufacturing and trade) is consistently embraced by PUNEW- and GMDC-ˆM∗
75%
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 35

THE MODEL CONFIDENCE SET 487
whether in levels or ﬁrst differences. In summary, we interpret these variables
as signals about the anticipated path of either real aggregate demand or real
aggregate supply that helps to predict inﬂation out of sample in the pre-1984
subsample.
There are several more inferences to draw from T ableIV. These concern the
two types of no-change forecasts whose predictive accuracy is strikingly differ-
ent. The no-change (month) forecast fails to appear in ˆM∗
75% either on the
pre-1984 or on the post-1984 subsamples, whereas the no-change (year) fore-
cast ﬁnds its way into ˆM∗
75% for the post-1984 subsample, but not the 1970:M1–
1983:M12 subsample. These results are especially of interest because the no-
change (year) forecast yields the best inﬂation forecasts on the 1984: M1–
1996:M9 subsample for both PUNEW and GMDC. These empirical results
for the no-change inﬂation forecasts are interesting because they reconcile the
results of Stock and Watson(1999) with those of Atkeson and Ohanian (2001).
Stock and Watson (1999, p. 327) found that “[T]he conventionally speciﬁed
Phillips curve, based on the unemployment rate, was found to perform rea-
sonably well. Its forecasts are better than univariate forecasting models (both
autoregressions and random walk models).” In contrast,Atkeson and Ohanian
(2001, p. 10) concluded that “economists have not produced a version of the
Phillips curve that makes more accurate inﬂation forecasts than those from a
naive model that presumes inﬂation over the next four quarters will be equal
to inﬂation over the last four quarters.” The source of the disagreement is that
Stock and Watson and Atkeson and Ohanian studied different no-change in-
ﬂation forecasts. The no-change forecast Stock and Watson (1999)d e p l o y e d
is last month’s inﬂation rate, whereas the no-change forecasts in Atkeson and
Ohanian (2001) is the past year’s inﬂation rate.
We agree with Stock and Watson that the Phillips curve is a device that yields
better forecasts of inﬂation in the pre-1984 period. The relevant ˆ
M∗
75% do not
include either of the no-change forecasts for PUNEW and GMDC. However,
for the post-1984 sample, we observe that no-change (year) forecast has the
smallest sample loss of all forecasts, which supports the conclusion of Atkeson
and Ohanian (2001).
Ta b l eV generates MCSs using factor models and forecast combination
methods that replicate the set of forecasts inStock and Watson(1999, T able 4).
They combined a large set of inﬂation forecasts from an array of 168 models
using sample means, sample medians, and ridge estimation to produce forecast
weighting schemes. The other forecasting approach depends on principal com-
ponents of the 168 macropredictors. The idea is that there exists an underlying
factor or factors (e.g., real aggregate demand, ﬁnancial conditions) that sum-
marize the information of a large set of predictors. For example, Solow (1976)
argued that a motivation for the Phillips curves of the 1960s and 1970s was that
unemployment captured, albeit imperfectly, the true unobserved state of real
aggregate demand.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 36

488 P . R. HANSEN, A. LUNDE, AND J. M. NASON
The factor models and forecast combination methods produce inﬂation fore-
casts that are, in general, better than those in T able IV. The forecasts con-
structed from “ All indicators” and “Real activity indicators” in Panels A and B
do particularly well across the board. Interestingly, the best forecast during the
1970:M1–1983:M12 subsample is the one-factor “ All indicators” model, while
the second best is the one-factor “Real activity indicators” model. Most of the
forecasts constructed from the “Money” variables do not ﬁnd their way into
the MCSs.
Despite the better predictive accuracy produced by factor models and fore-
cast combinations, during the post-1984 period the best forecast is the no-
change (year) forecast.
6.2. Likelihood-Based Comparison of T aylor-Rule Models
Monetary policy is often evaluated with the Ta y l o r(1993) rule. A T aylor rule
summarizes the objectives and constraints that deﬁne monetary policy by map-
ping (implicitly) from this decision problem to the path of the short-term nom-
inal interest rate. A canonical monetary policy loss function penalizes the deci-
sion maker for inﬂation volatility against its target and output volatility around
its trend. The mapping generates a T aylor rule that the interest rate responds
to inﬂation and output deviations from trend. Thus, T aylor rules measure ex
post the success monetary policy has had at meeting the goals of keeping inﬂa-
tion close to target and output at trend. Articles byTa y l o r(1999), Clarida, Galí,
and Gertler (2000), and Orphanides (2003) are leading examples of using T ay-
lor rules to evaluate actual monetary policy, while McCallum (1999)p r o v i d e d
an introduction for consumers of monetary policy rules.
This section shows how the MCS can be used to evaluate which T aylor rule
regression best approximates the underlying data generating process. We posit
the general T aylor rule regression
R
t =(1 − ρ)
[
γ0 +
pπ∑
j=1
γπ/commaorijπt−j +
py∑
j=1
γy/commaorijyt−j
]
+ρRt−1 +vt/commaori(5)
whereRt denotes the short-term nominal interest rate,πt is inﬂation,yt equals
deviations of output from trend (i.e., the output gap), and the error term,vt/commaoriis
assumed to be a martingale difference process. The T aylor principle is satisﬁed
if ∑pπ
j=1 γπ/commaorijexceeds 1 because a 1% rise in the sum of pπ lags of inﬂation indi-
cates thatRt should rise by more than 100 basis points. The monetary policy re-
sponse to real side ﬂuctuations is given by∑py
j=1 γy/commaorijon thepy lags of the output
gap. The interceptγ0 is the equilibrium steady state real rate plus the target in-
ﬂation rate (weighted by 1− ∑pπ
j=1 γπ/commaorij). The T aylor rule regression (5) includes
lagged interest, Rt−1/commaoriwhich may be interpreted as interest rate smoothing by
the central bank. Alternatively, the lagged interest rate could be interpreted as
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 37

THE MODEL CONFIDENCE SET 489
TABLE VI
TAYLOR RULE REGRESSION DATASETa
Observable Construction
Dependent variable
Rt : Interest rate Effective Fed Funds Rate (EFFR), T emporally aggregate daily
Rfed funds/commaorit return (annual rate) to quarterly,
Rt = 100 × ln[1 +Rfed funds/commaorit/100]
Independent variables
πt : Inﬂation Implicit GDP deﬂator, Pt , πt = 400 × ln[Pt/Pt−1]
seasonally adjusted (SA)
yt :O u t p u tg a p l n Qt − trendQt , i.e., transitory Apply Hodrick–Prescott ﬁlter
component of output, where Qt to lnQt
is real GDP in billions of chained
2000 $, SA at annual rates
urt : Unemployment UR t − trend URt , i.e., transitory T emporally aggregate monthly
rate gap component of UR t ,w h e r eU Rt is the to quarterly frequency to get UR t .
is the civilian unemployment rate, SA Apply Baxter–King ﬁlter to UR t
rulct : Real unit The cointegrating residual of nominal rulc t = LSt − LPt − ˆa0 − ˆa1t
labor costs ULC t (= LSt − LSt )a n dl nPt .L St is − ˆa2 lnPt
labor share, i.e., log of compensation
per hour in the nonfarm business
sector; LPt is labor productivity, i.e.,
log of output per hour of all persons
nonfarm business sector
aThe effective federal funds rate is obtained from H.15 Selected Interest Rates in Federal Reserve Statistical
Releases. The implicit price deﬂator, real GDP , the unemployment rate, compensation per hour, and output per hour
of all persons are constructed by the Bureau of Economic Analysis and are available at the FRED Data Bank at the
Federal Reserve Bank of St. Louis. The sample period is 1979: Q1–2006:Q4. The data are drawn from data available
online from the Board of Governors and FRED at the Federal Reserve Bank of St. Louis.
a proxy for other determinants of the interest rate that are not captured by the
regression (5). Note also that the T aylor rule regression (5) avoids issues that
arise in the estimation of simultaneous equation systems because contempora-
neous inﬂation,πt , and the output gap,yt , are not regressors, only lags of these
variables are. In this case, structural interpretations have to be applied to the
T aylor rule regression (5)w i t hc a r e .
The T aylor rule regression (5) is estimated by ordinary least squares on a
U.S. sample that runs from 1979:Q1 to 2006:Q4. T ableVI provides details about
the data used to estimate the T aylor rule regression.15 The (effective) federal
funds rate deﬁnes the T aylor rule policy rate Rt . The growth rate of the im-
15We have generated results on a shorter post-1984 sample. Omitting the volatile 1979–1983
period from the analysis does not substantially change our results, beyond the loss of information
that one would expect with a shorter sample. These results are available in a separate appendix
(Hansen, Lunde, and Nason (2011)).
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 38

490 P . R. HANSEN, A. LUNDE, AND J. M. NASON
plicit gross domestic product (GDP) deﬂator is our measure of inﬂation, πt .
The cyclical component of the Hodrick and Prescott (1997) ﬁlter is applied to
real GDP to obtain estimates of the output gap, yt . We also employ two real
activity variables to ﬁll out the model space and to act as alternatives to the out-
put gap. These real activity variables are the Baxter and King (1999) ﬁltered
unemployment rate gap, urt , and the Nason and Smith (2008) measure of real
unit labor costs, rulc t . We compute the Baxter–King ur t using the maximum
likelihood–Kalman ﬁlter methods of Harvey and T rimbur(2003).
The model space consists of 25 speciﬁcations. The model space is built by
setting ρto zero or estimating it ( pπ = 1o r2 /commaoripy = 1 or 2) and equating yt
with the output gap, or replacing it with either the unemployment rate gap or
real unit labor costs. We add to these 24 ( = 2 × 2 × 3 × 2) regressions a pure
AR(1) model of the effective federal funds rate.
TABLE VII
MCS FOR TAYLOR RULES: 1979:Q1–2006:Q4a
Model Speciﬁcation Q(Zj/commaoriˆθj) ˆk⋆ KLIC AIC ⋆ BIC⋆
Rt−1 93.15 13.74 106.89 (0.30) ∗∗ 120.63 (0.47)∗∗ 157.99 (0.63)∗∗
πt−1 yt−1 284.82 11.44 296.25 (0.00) 307.69 (0.00) 338.79 (0.00)
πt−j ,j=1/commaori2 yt−j ,j=1/commaori2 258.95 14.66 273.61 (0.00) 288.28 (0.01) 328.14 (0.01)
πt−1 urt−1 289.65 10.20 299.84 (0.00) 310.04 (0.00) 337.75 (0.00)
πt−j ,j=1/commaori2 urt−j ,j=1/commaori2 268.90 12.82 281.72 (0.00) 294.53 (0.00) 329.37 (0.01)
πt−1 rulct−1 289.99 9.89 299.88 (0.00) 309.77 (0.00) 336.67 (0.01)
πt−j ,j=1/commaori2 rulct−j ,j=1/commaori2 266.07 12.12 278.19 (0.00) 290.31 (0.01) 323.26 (0.01)
yt−1 urt−1 387.45 17.04 404.49 (0.00) 421.54 (0.00) 467.86 (0.00)
yt−j , j=1/commaori2 urt−j ,j=1/commaori2 385.86 23.42 409.28 (0.00) 432.69 (0.00) 496.35 (0.00)
yt−1 rulct−1 386.47 14.92 401.39 (0.00) 416.32 (0.00) 456.89 (0.00)
yt−j , j=1/commaori2 rulct−j ,j=1/commaori2 385.43 19.44 404.87 (0.00) 424.31 (0.00) 477.16 (0.00)
urt−1 rulct−1 386.21 15.41 401.62 (0.00) 417.02 (0.00) 458.90 (0.00)
urt−j ,j=1/commaori2 rulct−j ,j=1/commaori2 384.82 19.86 404.68 (0.00) 424.54 (0.00) 478.52 (0.00)
Rt−1 πt−1 yt−1 68.57 17.71 86.28 (0.86) ∗∗ 103.98 (1.00)∗∗ 152.12 (0.64)∗∗
Rt−1 πt−j ,j=1/commaori2 yt−j ,j=1/commaori2 62.11 22.11 84.22 (1.00) ∗∗ 106.32 (0.93)∗∗ 166.43 (0.41)∗∗
Rt−1 πt−1 urt−1 77.57 16.32 93.89 (0.72) ∗∗ 110.22 (0.89)∗∗ 154.60 (0.64)∗∗
Rt−1 πt−j ,j=1/commaori2 urt−j ,j=1/commaori2 73.27 18.79 92.07 (0.80) ∗∗ 110.86 (0.89)∗∗ 161.95 (0.57)∗∗
Rt−1 πt−1 rulct−1 72.80 16.06 88.86 (0.86) ∗∗ 104.92 (0.93)∗∗ 148.58 (1.00)∗∗
Rt−1 πt−j ,j=1/commaori2 rulct−j ,j=1/commaori2 69.21 19.26 88.47 (0.86) ∗∗ 107.73 (0.92)∗∗ 160.09 (0.58)∗∗
Rt−1 yt−1 urt−1 86.16 19.16 105.33 (0.33) ∗∗ 124.49 (0.38)∗∗ 176.59 (0.16)∗
Rt−1 yt−j , j=1/commaori2 urt−j ,j=1/commaori2 85.51 24.32 109.83 (0.28) ∗∗ 134.16 (0.18)∗ 200.28 (0.02)
Rt−1 yt−1 rulct−1 89.42 18.92 108.35 (0.29) ∗∗ 127.27 (0.31)∗∗ 178.72 (0.15)∗
Rt−1 yt−j , j=1/commaori2 rulct−j ,j=1/commaori2 88.11 22.42 110.53 (0.28) ∗∗ 132.94 (0.20)∗ 193.88 (0.03)
Rt−1 urt−1 rulct−1 87.42 18.07 105.49 (0.33) ∗∗ 123.55 (0.38)∗∗ 172.66 (0.21)∗
Rt−1 urt−j ,j=1/commaori2 rulct−j ,j=1/commaori2 85.93 21.32 107.25 (0.30) ∗∗ 128.56 (0.28)∗∗ 186.51 (0.06)
aWe report the maximized log-likelihood function (multiplied by − 2), the effective degress of freedom, and the
three criteria KLIC, AIC⋆, and BIC⋆ along with the corresponding MCS p-values. The regression models in ˆM∗
90%
and ˆM∗
75%
are identiﬁed by one and two asterisks, respectively. See the text and T able VI f o rv a r i a b l em n e m o n i c s
and deﬁnitions.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 39

THE MODEL CONFIDENCE SET 491
TABLE VIII
REGRESSION MODELS IN ˆM∗
90%-KLICa
γ0 ργ π/commaori1 γπ/commaori2 γy/commaori1 γy/commaori2 γur/commaori1 γur/commaori2 γrulc/commaori1 γrulc/commaori2
5.29 0.96
(2.50) (30.1)
0.12 0.84 1.87 1.20
(0.13) (17.0) (7.01) (2.17)
0.00 0.80 0.77 1.14 1.50 −0.39
(0.00) (12.1) (2.58) (4.76) (1.25) (0.33)
0.82 0.86 1.60 1.58
(0.67) (16.8) (4.85) (0.25)
0.64 0.83 0.68 0.97 5.90 −6.56
(0.56) (12.9) (1.77) (2.85) (0.68) (1.16)
0.37 0.87 1.76 −0.81
(0.30) (17.0) (5.38) (1.56)
0.39 0.84 0.76 0.99 −0.18 −0.55
(0.35) (12.9) (2.12) (3.55) (0.23) (0.68)
5.63 0.97 4.89 45.9
(2.20) (37.3) (1.05) (0.79)
5.56 0.97 6.42 −1.71 60.7 −22.9
(2.12) (32.3) (0.58) (0.19) (0.66) (0.42)
5.33 0.97 1.04 −2.47
(2.22) (35.5) (0.32) (0.79)
5.42 0.97 8.37 −8.05 2.52 −5.43
(2.22) (32.6) (0.64) (0.56) (0.75) (0.96)
5.35 0.97 30.9 −3.62
(2.02) (37.8) (0.63) (1.04)
5.43 0.97 52.5 −25.6 −1.18 −2.74
(2.10) (34.2) (0.64) (0.54) (0.30) (0.85)
aParameter estimates witht-statistics (in absolute values) in parentheses. The shaded area identiﬁes the models in
ˆM∗
75%-BIC⋆.
We present results of applying the MCS and likelihood-based criteria to the
choice of the best T aylor rule regression ( 5) and AR(1) regressions in T a-
bles VII and VIII.T a b l eVII reports Q(Zj/commaoriˆθj) (the log-likelihood function
multiplied by −2), the bootstrap estimate of the effective degrees of free-
dom, ˆk⋆, and the realizations of the three empirical criteria, KLIC, AIC ⋆,
and BIC⋆. The numbers surrounded by parentheses in columns headed KLIC,
AIC⋆,a n dB I C⋆ are the MCS p-values, and an asterisk identiﬁes the speciﬁ-
cations that enter ˆM∗
90%.T a b l eVIII lists estimates of the regression models
that are in ˆM∗
90%
along with their corresponding t-statistics in parentheses.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 40

492 P . R. HANSEN, A. LUNDE, AND J. M. NASON
The t-statistics are based on robust standard errors following Newey and West
(1987).
Ta b l eVII shows that the MCS procedure selects 10–13 of the 25 possible
regressions depending on the information criteria. The lagged nominal rate
Rt−1 is the one regressor common to the regressions that enter ˆM∗
90% for the
KLIC, AIC⋆,a n dB I C⋆. Besides the AR(1), ˆM∗
90%
consists of the six T aylor rule
speciﬁcations that nest the AR(1). Under the KLIC and AIC⋆, the T aylor rule
regressions include all one or two lag combinations of πt , yt ,u rt , and rulc t .
The BIC produces a smaller ˆM∗
90% because it ejects the two lag T aylor rule
speciﬁcations that exclude lagged πt . Thus, the T aylor rule regression–MCS
example ﬁnds that the BIC tends to settle on more parsimonious models. This
is to be expected, given its larger penalty on model complexity.
The AR(1) falls into ˆM∗
90% under the KLIC, AIC⋆,a n dB I C⋆. Although the
ﬁrst line of T able VII shows that the AR(1) has the largest Q(Zj/commaoriˆθj) of the
regressions covered by ˆM∗
90%
, the MCS recruits the AR(1) because it has a
relatively small estimate of the effective degrees of freedom,ˆk⋆/periodoriIt is important
to keep in mind that estimates of the effective degrees of freedom are larger
than the number of free parameters in each of the models. This reﬂects the
fact that the Gaussian model is misspeciﬁed. For example, the conventional
AIC penalty (that doubles the number of free parameters) is misleading in the
context of misspeciﬁed models; seeTa k e u c h i(1976), Sin and White(1996), and
Hong and Preston (2008).
It is somewhat disappointing that the MCS procedure yields as many as 13
models in ˆ
M∗
90%. The reason is that the data lack the information to resolve
precisely which T aylor rule speciﬁcation is best in terms of Kullback–Leibler
discrepancy. The large set of models is also an outcome of the strict require-
ments that characterize the MCS. The MCS procedure is designed to control
the familywise error rate (FWE), which is the probability of making one or
more false rejections. We will be able to trim ˆ
M∗ further if we relax the con-
trol of the FWE, but that will affect the interpretation of ˆM∗
1−α. For instance,
if we control the probability of makingk or more false rejections,k-FWE (see,
e.g., Romano, Shaikh, and Wolf (2008)), additional models can be eliminated.
The drawback ofk-FWE and other alternative controls is that the MCS looses
its key property, which is to contain the best models with probability 1 − α/periodori
Ta b l eVIII provides information about the regressions in ˆM∗
90%-KLIC. The
shaded area identiﬁes the models in ˆM∗
75%
-BIC⋆/periodoriFirst, note that the estimated
T aylor rules always satisfy the T aylor principle (i.e.,ˆγπ/commaori1 > 1o r ˆγπ/commaori1 +ˆγπ/commaori2 > 1).
The coefﬁcients associated with real activity variables have insigniﬁcant t-
statistics in most cases. Only the ﬁrst lag of the output gap produces a positive
coefﬁcient with a t-ratio above 2 in the ﬁrst T aylor rule regression listed in T a-
ble VIII. Moreover, the statistically insigniﬁcant coefﬁcients for the unemploy-
ment rate gap and real unit labor costs variables often have counterintuitive
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 41

THE MODEL CONFIDENCE SET 493
signs. Finally, the estimates of ρare between 0.83 and 0.87 in the T aylor rule
regressions that include a lag of πt , which suggests interest rate smoothing.16
The fact that the MCS cannot settle on a single speciﬁcation is not a sur-
prising result. Monetary policymakers almost surely rely on a more complex
information set than can be summarized by a simple model. Furthermore, any
real activity variable is an imperfect measure of the underlying state of the
economy, and there are important and unresolved issues regarding the mea-
surement of gap and marginal cost variables that translate into uncertainty
about the proper deﬁnitions of the real activity variables.
7.
SUMMAR Y AND CONCLUDING REMARKS
This paper introduces the model conﬁdence set (MCS) procedure, relates it
to other approaches of model selection and multiple comparisons, and estab-
lishes the asymptotic theory of the MCS. The MCS is constructed from a hy-
pothesis test,δ
M/commaoriand an elimination rule,eM/periodoriWe deﬁned coherency between
test and elimination rule, and stressed the importance of this concept for the
ﬁnite sample properties of the MCS. We also outlined simple and convenient
bootstrap methods for the implementation of the MCS procedure. The paper
employs Monte Carlo experiments to study the MCS procedure that reveal it
has good small sample properties.
It is important to understand the principle of the MCS procedure in applica-
tions. The MCS is constructed such that inference about the “best” follows the
conventional meaning of the word “signiﬁcance.” Although the MCS will con-
tain only the best model(s) asymptotically, it may contain several poor models
in ﬁnite samples. A key feature of the MCS procedure is that a model is dis-
carded only if it is found to be signiﬁcantly inferior to another model. Models
remain in the MCS until proven inferior, which has the implication that not all
models in the MCS may be judged good models.
17
An important advantage of the MCS, compared to other selection proce-
dures, is that the MCS acknowledges the limits to the informational content
of the data. Rather than selecting a single model without regard to degree of
information, the MCS procedure yields a set of models that summarizes key
sample information.
We applied the MCS procedure to the inﬂation forecasting problem ofStock
and Watson(1999). Results show that the MCS procedure provides a powerful
tool for evaluating competing inﬂation forecasts. We emphasize that the infor-
mation content of the data matters for the inferences that can be drawn. The
16We have also estimated T aylor rule regressions with moving average (MA) errors, as an al-
ternative to using Rt−1 as a regressor. The empirical ﬁt of models with MA errors is, in all cases,
inferior to the T aylor rule regressions that includeRt−1/periodori
17The proportion of models in ˆM∗
1−α that are members of M∗ can be related to the false
discovery rate and the q-value theory of Storey (2002). See McCracken and Sapp (2005)f o ra n
application that compares forecasting models. See also Romano, Shaikh, and Wolf (2008).
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 42

494 P . R. HANSEN, A. LUNDE, AND J. M. NASON
great inﬂation–disinﬂation subsample of 1970:M1–1983:M12 has movements in
inﬂation and macrovariables that allow the MCS procedure to make relatively
sharp choices across the relevant models. The information content of the less
persistent, less volatile 1984: M1–1996:M9 subsample is limited in comparison
because the MCS procedure lets in almost any model that Stock and Watson
considered. A key exception is the no-change (month) forecast that uses last
month’s inﬂation rate as a predictor of future inﬂation. This no-change fore-
cast never resides in the MCS in either the earlier or the later periods. A likely
explanation is that month-to-month inﬂation is a noisy measure of core inﬂa-
tion. This view is supported by the fact that a second no-change (year) fore-
cast, which employs a year-over-year inﬂation rate as the forecast, is a better
forecast. This result enables us to reconcile the empirical results in Stock and
Watson (1999) with those of Atkeson and Ohanian (2001). Nonetheless, the
question of what constitutes the best inﬂation forecasting model for the last
35 years of U.S. data remains unanswered because the data provide insufﬁ-
cient information to distinguish between good and bad models.
This paper also constructs a MCS for T aylor rule regressions based on three
likelihood criteria. Such interest rate rules are often used to evaluate the suc-
cess of monetary policy, but this is not our intent for the MCS. Instead, we
study the MCS that selects the best ﬁtting T aylor rule regressions under either
a quasi-likelihood criterion, the AIC, or the BIC using the effective degrees of
freedom. The competing T aylor rule regressions consist of different combina-
tions of lags of inﬂation, lags of three different real activity variables, and the
lagged federal funds rate. Besides these T aylor rule regressions, the MCS must
also contend with a ﬁrst-order autoregression of the federal funds rate. The
regressions are estimated on a 1979:
Q1–2006:Q4 sample of U.S. data. Under
the three likelihood criteria, the MCS settles on T aylor rule regressions that
satisfy the T aylor principle, include all three competing real activity variables,
and add the lagged federal funds rate. Furthermore, we ﬁnd that the ﬁrst-order
autoregression also enters the MCS. Thus, the U.S. data lack the information
to resolve precisely which T aylor rule speciﬁcation best describes the data.
Given the large number of forecasting problems economists face at central
banks and other parts of government, in ﬁnancial markets, and other settings,
the MCS procedure faces a rich set of problems to study. Furthermore, the
MCS has a wide variety of potential uses beyond forecast comparisons and
regression models. We leave this work for future research.
REFERENCES
ANDERSON, T . W . (1984):An Introduction to Multivariate Statistical Analysis (Second Ed.). New
Yo r k : Wi l e y .[455]
ANDREWS, D. W . K. (1991): “Heteroskedasticity and Autocorrelation Consistent Covariance Ma-
trix Estimation,” Econometrica, 59, 817–858. [470]
ATKESON,A . ,AND L. E. OHANIAN (2001): “ Are Phillips Curves Useful for Forecasting Inﬂation?”
Federal Reserve Bank of Minneapolis Quarterly Review , 25, 2–11. [456,483,484,487,494]
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 43

THE MODEL CONFIDENCE SET 495
BAXTER,M . ,AND R. G. KING (1999): “Measuring Business Cycles: Approximate Bandpass Fil-
ters for Economic Time Series,” Review of Economics and Statistics , 81, 575–593. [490]
BERGER, R. L. (1982): “Multiparameter Hypothesis T esting and Acceptance Sampling,”T echno-
metrics, 24, 295–300. [473]
BERNANKE,B .S . ,AND J. BOIVIN (2003): “Monetary Policy in a Data-Rich Environment,”Journal
of Monetary Economics, 50, 525–546. [457]
CAVANAUGH,J .E . , AND R. H. SHUMWAY (1997): “ A Bootstrap V ariant of AIC for State-Space
Model Selection,” Statistica Sinica, 7, 473–496. [471]
CHAO,J .C . ,V .C ORRADI, AND N. R. S WANSON (2001): “ An Out of Sample T est for Granger
Causality,” Macroeconomic Dynamics, 5, 598–620. [476]
CHONG,Y .Y . ,AND D. F . HENDR Y(1986): “Econometric Evaluation of Linear Macroeconomic
Models,” Review of Economic Studies , 53, 671–690. [476]
CLARIDA,R . ,J .G ALÍ, AND M. GERTLER (2000): “Monetary Policy Rules and Macroeconomic
Stability: Evidence and Some Theory,” Quarterly Journal of Economics , 115, 147–180. [488]
CLARK,T .E . ,AND M. W . MCCRACKEN (2001): “T ests of Equal Forecast Accuracy and Encom-
passing for Nested Models,” Journal of Econometrics, 105, 85–110. [475,476]
(2005): “Evaluating Direct Multi-Step Forecasts,” Econometric Reviews , 24, 369–404.
[466]
DIEBOLD,F .X . ,AND R. S. MARIANO (1995): “Comparing Predictive Accuracy,” Journal of Busi-
ness & Economic Statistics , 13, 253–263. [465]
DOORNIK, J. A. (2009): “ Autometrics,” in The Methodology and Practice of Econometrics:
A Festschrift in Honour of David F . Hendry , ed. by N. Shephard and J. L. Castle. New Y ork:
Oxford University Press, 88–121. [468]
(2006): Ox: An Object-Orientated Matrix Programming Language (Fifth Ed.). London:
Timberlake Consultants Ltd. [453]
DUDOIT,S . ,J .P .SHAFFER, AND J. C. BOLDRICK (2003): “Multiple Hypothesis T esting in Microar-
ray Experiments,” Statistical Science, 18, 71–103. [473,474]
EFRON, B. (1983): “Estimating the Error Rate of a Prediction Rule: Improvement on Cross-
V alidation,”Journal of the American Statistical Association , 78, 316–331. [470,471]
(1986): “How Biased Is the Apparent Error Rate of a Prediction Rule?” Journal of the
American Statistical Association, 81, 461–470. [471]
ENGLE,R .F . ,AND S. J. BROWN (1985): “Model Selection for Forecasting,” Journal of Computa-
tion in Statistics, 51, 341–365. [475]
GIACOMINI,R . ,AND H. WHITE (2006): “T ests of Conditional Predictive Ability,”Econometrica,
74, 1545–1578. [476,484]
GONCALVES,S . ,AND H. WHITE (2005): “Bootstrap Standard Error Estimates for Linear Regres-
sion,” Journal of the American Statistical Association , 100, 970–979. [468,470]
GORDON, R. J. (1997): “The Time-V arying NAIRU and Its Implications for Economic Policy,”
Journal of Economic Perspectives, 11, 11–32. [456]
GRANGER, C. W . J., M. L. KING, AND H. WHITE (1995): “Comments on T esting Economic The-
ories and the Use of Model Selection Criteria,” Journal of Econometrics, 67, 173–187. [475]
GUPTA,S .S . , AND S. PANCHAP AKESAN(1979): Multiple Decision Procedures . New Y ork: Wiley.
[473]
HANSEN, P . R. (2003a): “ Asymptotic T ests of Composite Hypotheses,” Working Paper 03-09,
Brown University Economics. Available at http://ssrn.com/abstract=399761.[ 475]
(2003b): “Regression Analysis With Many Speciﬁcations: A Bootstrap Method to Ro-
bust Inference,” Mimeo, Stanford University. [466]
(2005): “ A T est for Superior Predictive Ability,”Journal of Business & Economic Statis-
tics, 23, 365–380. [466,471,474]
HANSEN,P .R . ,A .L UNDE, AND J. M. N ASON (2011): “Supplement to ‘The Model Con-
ﬁdence Set’,” Econometrica Supplemental Material , 79, http://www.econometricsociety.org/
ecta/Supmat/5771_tables.pdf; http://www.econometricsociety.org/ecta/Supmat/5771_data and
programs.zip.[ 457,467,481,484,489]
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 44

496 P . R. HANSEN, A. LUNDE, AND J. M. NASON
HARVEY,A .C . ,AND T. M . TRIMBUR (2003): “General Model-Based Filters for Extracting Cycles
and T rends in Economic Time Series,”Review of Economics and Statistics , 85, 244–255. [490]
HARVEY, D., AND P. NEWBOLD (2000): “T ests for Multiple Forecast Encompassing,” Journal of
Applied Econometrics, 15, 471–482. [476]
HODRICK,R .J . ,AND E. C. PRESCOTT (1997): “Postwar U.S. Business Cycles: An Empirical In-
vestigation,” Journal of Money, Credit, and Banking Economy , 29, 1–16. [484,490]
HONG,H . ,AND B. PRESTON (2008): “Bayesian Averaging, Prediction and Nonnested Model Se-
lection,” Working Paper W14284, NBER. [470,492]
HORRACE,W .C . ,AND P. SCHMIDT (2000): “Multiple Comparisons With the Best, With Economic
Applications,” Journal of Applied Econometrics , 15, 1–26. [473]
HSU, J. C. (1996): Multiple Comparisons. Boca Raton, FL: Chapman & Hall/CRC. [473]
INOUE,A . ,AND L. KILIAN (2006): “On the Selection of Forecasting Models,” Journal of Econo-
metrics, 130, 273–306. [475]
JOHANSEN, S. (1988): “Statistical Analysis of Cointegration Vectors,” Journal of Economic Dy-
namics and Control, 12, 231–254. [455,473]
KILIAN, L. (1999): “Exchange Rates and Monetary Fundamentals: What Do We Learn From
Long Horizon Regressions?” Journal of Applied Econometrics , 14, 491–510. [466]
LEEB,H . ,AND B. PÖTSCHER (2003): “The Finite-Sample Distribution of Post-Model-Selection
Estimators, and Uniform versus Non-Uniform Approximations,” Econometric Theory , 19,
100–142. [460]
LEHMANN,E .L . ,AND J. P . ROMANO (2005): T esting Statistical Hypotheses(Third Ed.). New Y ork:
Wiley. [464,473,474]
MCCALLUM, B. T . (1999): “Issues in the Design of Monetary Policy Rules,” in Handbook of
Macroeconomics, Vol. 1C, ed. by J. B. T aylor and M. Woodford. Amsterdam: North-Holland,
1483–1530. [488]
MCCRACKEN,M .W . ,AND S. SAPP (2005): “Evaluating the Predictability of Exchange Rates Using
Long Horizon Regressions: Mind Y ourp’s andq’s!” Journal of Money, Credit, and Banking, 37,
473–494. [493]
NASON,J .M . ,AND G. W . SMITH (2008): “Identifying the New Keynesian Phillips Curve,” Journal
of Applied Econometrics, 23, 525–551. [490]
NEWEY,W . ,AND K. WEST (1987): “ A Simple Positive Semi-Deﬁnite, Heteroskedasticity and Au-
tocorrelation Consistent Covariance Matrix,” Econometrica, 55, 703–708. [464,470,492]
ORPHANIDES, A. (2003): “Historical Monetary Policy Analysis and the T aylor Rule,” Journal of
Monetary Economics, 50, 983–1022. [488]
ORPHANIDES,A . ,AND S. VAN NORDEN (2002): “The Unreliability of Output-Gap Estimates in
Real Time,” Review of Economics and Statistics , 84, 569–583. [457]
PANTULA, S. G. (1989): “T esting for Unit Roots in Time Series Data,” Econometric Theory ,5 ,
256–271. [473]
ROMANO,J .P . ,AND M. WOLF (2005): “Stepwise Multiple T esting as Formalized Data Snooping,”
Econometrica, 73, 1237–1282. [474]
ROMANO,J .P . ,A .M .S HAIKH, AND M. WOLF (2008): “Formalized Data Snooping Based on
Generalized Error Rates,” Econometric Theory, 24, 404–447. [474,492,493]
SHIBATA, R. (1997): “Bootstrap Estimate of Kullback–Leibler Information for Model Selection,”
Statistica Sinica, 7, 375–394. [471]
SHIMODAIRA, H. (1998): “ An Application of Multiple Comparison T echniques to Model Selec-
tion,” Annals of the Institute of Statistical Mathematics , 50, 1–13. [473]
SIN,C . - Y . ,AND H. WHITE (1996): “Information Criteria for Selecting Possibly Misspeciﬁed Para-
metric Models,” Journal of Econometrics, 71, 207–225. [468,470,475,492]
SOLOW, R. M. (1976): “Down the Phillips Curve With Gun and Camera,” in I n ﬂ a t i o n ,T r a d e ,a n d
Ta x e s, ed. by D. A. Belsley, E. J. Kane, P . A. Samuelson, and R. M. Solow. Columbus, OH: Ohio
State University Press. [487]
STAIGER, D., J. H. STOCK, AND M. W . WATSON (1997a): “How Precise Are Estimates of the Nat-
ural Rate of Unemployment?” in Reducing Inﬂation: Motivation and Strategy ,e d .b yC .R o m e r
and D. Romer. Chicago: University of Chicago Press, 195–242. [457]
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License


## Page 45

THE MODEL CONFIDENCE SET 497
(1997b): “The NAIRU, Unemployment, and Monetary Policy,” Journal of Economic
Perspectives, 11, 33–49. [456]
STOCK,J .H . , AND M. W . WATSON (1999): “Forecasting Inﬂation,” Journal of Monetary Eco-
nomics, 44, 293–335. [453-456,483,484,487,493,494]
(2003): “Forecasting Output and Inﬂation: The Role of Asset Prices,” Journal of Eco-
nomic Literature, 61, 788–829. [456]
STOREY, J. D. (2002): “ A Direct Approach to False Discovery Rates,”Journal of the Royal Statis-
tical Society, Ser. B, 64, 479–498. [493]
TAKEUCHI, K. (1976): “Distribution of Informational Statistics and a Criterion of Model Fitting,”
Suri-Kagaku (Mathematical Sciences), 153, 12–18. (In Japanese.) [470,492]
TAYLOR, J. B. (1993): “Discretion versus Policy Rules in Practice,”Carnegie–Rochester Conference
Series on Public Policy, 39, 195–214. [456,488]
(1999): “ A Historical Analysis of Monetary Policy Rules,” inMonetary Policy Rules,e d .
by J. B. T aylor. Chicago: University of Chicago Press, 319–341. [488]
VUONG, Q. H. (1989): “Likelihood Ratio T ests for Model Selection and Non-Nested Hypothe-
ses,” Econometrica, 57, 307–333. [471,473]
WEST, K. D. (1996): “ Asymptotic Inference About Predictive Ability,” Econometrica, 64,
1067–1084. [465]
WEST,K .D . , AND D. CHO (1995): “The Predictive Ability of Several Models of Exchange Rate
Volatility,” Journal of Econometrics, 69, 367–391. [464]
WEST, K. D., AND M. W . MCCRACKEN (1998): “Regression Based T ests of Predictive Ability,”
International Economic Review, 39, 817–840. [475]
WHITE, H. (1994): Estimation, Inference and Speciﬁcation Analysis . Cambridge: Cambridge Uni-
versity Press. [469]
(2000a): Asymptotic Theory for Econometricians (Revised Ed.). San Diego: Academic
Press. [464]
(2000b): “ A Reality Check for Data Snooping,”Econometrica, 68, 1097–1126. [455,466,
471,474]
Dept. of Economics, Stanford University, 579 Serra Mall, Stanford, CA 94305-
6072, U.S.A. and CREATES; peter.hansen@stanford.edu,
School of Economics and Management, Aarhus University, Bartholins Allé 10,
Aarhus, Denmark and CREATES; alunde@econ.au.dk,
and
Federal Reserve Bank of Philadelphia, T en Independence Mall, Philadelphia,
PA 19106-1574, U.S.A.; Jim.Nason@phil.frb.org.
Manuscript received March, 2005; ﬁnal revision received March, 2010.
 14680262, 2011, 2, Downloaded from https://onlinelibrary.wiley.com/doi/10.3982/ECTA5771 by Suny Binghamton, Wiley Online Library on [08/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
