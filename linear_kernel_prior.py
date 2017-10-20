import GPy
import numpy as np

def get_pred_with_linear_prior(m,priormeans,priorvariances,predXs,lindim):
    """
    Generate a prediction for a GPRegression model with a kernel of the 
    form otherkern * linear
    
    m = the GPy model
    priormeans = a list or 1d numpy array of the priors on the gradient
                 of the linear kernel at each of the prediction point.
    priorvariances = a list or 1d numpy array of the uncertainty of the
                 priors.
    predXs = the test inputs, as a numpy array NxD
    lindim = the dimension of the inputs which is the linear kernel.
    
    Returns:
    m_w_prior = model with prior
    preds = mean prediction
    predvars = predicted variances
    
    Example:
    
        X = np.array([[3,1],[5,2],[4,3],[6,4]])
        Y = np.array([3,5,4,6])
        kern = GPy.kern.RBF(1,active_dims=[1],lengthscale=40.0) * GPy.kern.Linear(1,active_dims=[0])
        m = GPy.models.GPRegression(X,Y[:,None],kern)
        priormeans = [2.0,2.0,2.0,4.0]
        priorvariances = [0.1,0.1,0.1,0.000001]
        predXs = np.array([[5,3],[10,3],[5,60],[5,60]])
        
        mcopy,ps,pvars = get_pred_with_linear_prior(m,priormeans,priorvariances,predXs,0)
        print(ps)
        print(pvars)
    
    Output:
    
        [5.4659524887, 10.931904977, 9.1407971685, 19.999978720]
        [0.2579967643, 1.0319870572, 2.2384737322, 2.4999995773e-05]
    """
    
    assert len(priormeans)==len(predXs)
    assert len(priormeans)==len(priorvariances)    
    preds = []
    predvars = []
    for predX, priormean, priorvariance in zip(predXs, priormeans, priorvariances):
        tempX = 1000000.0
        inducingX = predX.copy()
        inducingX[lindim] = tempX
        inducingX = inducingX[None,:]
        #inducingX = np.array([[predX[0],tempX]])
        newX = np.r_[m.X,inducingX]
        newY = np.r_[m.Y[:,0],priormean*tempX]
        variances = np.ones(newX.shape[0])*0.0001
        variances[-1] = tempX**2 * priorvariance
        kern_w_priornoise = m.kern + GPy.kern.WhiteHeteroscedastic(1,len(newX),variances,active_dims=[lindim])
        m_w_prior = GPy.models.GPRegression(newX,newY[:,None],kern_w_priornoise)
        pred, predvar = m_w_prior.predict_noiseless(predX[None,:])
        preds.append(pred[0,0])
        predvars.append(predvar[0,0])
    return m_w_prior,preds,predvars
