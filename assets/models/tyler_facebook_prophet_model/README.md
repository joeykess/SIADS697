## Facebook Prophet Model Exploration

#### Intent:
<p>
This modeling process was meant to test techniques that approximate a target price of multiple stocks in the S&P 500 at some point in the future. Facebook's Prophet library was used in an exploratory fashion to understand how the algorithm works, but not used for this project.
</p>

#### Data Used:
1. Historical stock prices (from 1980 forward) for every stock in the S&P 500
2. Data was truncated to 2016 forward to account for space and performance limitations
3. Data features were created using the Technical Analysis python library, but we also created our own functions to create technical features

#### Future Consideration:
<p> Prophet could be used to create economic cycle features that help understand the general direction of the market. This can be quite useful as it opens the prediction process up to more than just technical feature representation</b>
