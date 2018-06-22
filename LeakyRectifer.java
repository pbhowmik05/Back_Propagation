
public class LeakyRectifer extends Layer  {

	LeakyRectifer(int inputs) {
		super(inputs, inputs);		
	}

@Override
    public void activate(Vec weights, Vec x) {
   //   System.out.println("LeakyRectifer.activate(Vec weights, Vec x) is executing" );
        for(int i = 0; i < activation.len; i++) {
            double value_X = x.get(i);
            activation.set(i, x.get(i)>0 ? value_X: 0.01*value_X );
        }
    }

@Override
    public void backProp(Vec weights, Vec prevBlame) {
    //System.out.println("LeakyRectifer.backProp() is executing");

        for(int i = 0; i < prevBlame.len; i++) {
            double activation_value = activation.get(i);
		prevBlame.set(i, activation_value<0? 0.01:1);

	}
    }
@Override
	public void updateGradient(Vec x, Vec gradient) {
       //     System.out.println("LeakyRectifer.updateGradient() is executing");
		
	}
}
