/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author pankaj
 */

abstract class Layer
{
    protected Vec activation;
    protected int mInput_size;
    protected int mOutput_size;
    protected Vec blame;
    
    Layer(int inputs, int outputs)
    {
        mInput_size = inputs;
        mOutput_size = outputs;
        activation = new Vec(outputs);   
        blame = new Vec(outputs);
    }
    
    abstract void activate(Vec weights, Vec x);
    abstract void backProp(Vec weights, Vec prevBlame);
    abstract void updateGradient(Vec x, Vec gradient);  
}
     




 


