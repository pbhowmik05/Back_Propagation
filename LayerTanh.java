/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author pankaj
 */
////////////////////////////////////////////////////////////////////////////////
public class LayerTanh extends Layer {

    LayerTanh(int inputs) {
        super(inputs, inputs);
    }
	
    @Override
    void activate(Vec weights, Vec x) {
        for(int i = 0; i < activation.len; i++){		
            double act_val = Math.tanh(x.get(i));
            activation.set(i, act_val);
        }    
    }

    @Override
    void backProp(Vec weights, Vec prevBlame) {
        for(int i = 0; i < prevBlame.len; i++)	{
            double grad= activation.get(i);
            prevBlame.set(i, blame.get(i)*(1.0 - Math.pow(grad, 2)));		
	}
    }
    @Override
    void updateGradient(Vec x, Vec gradient) {		
    }
}