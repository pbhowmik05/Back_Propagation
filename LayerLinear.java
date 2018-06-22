import java.util.Random;
public class LayerLinear extends Layer
{
    LayerLinear(int inputs, int outputs)
    {
        super(inputs, outputs);
    }
	protected double[] vals;
	protected int start;
        protected int len;
        
    public int size() {
        return len;
    }    
    
    @Override
    void activate(Vec weights, Vec x)
    {
        //weights = new Vec(new double[]{0.1,0.2,0.3,0.4,0.1,0.2,0.3});
       // Vec M = new Vec(new double[]{0.1,0.2,0.3,0.4});
        //Vec b = new Vec(new double[]{0.1,0.2,0.3});
        activation.fill(0);
        Vec M = new Vec(weights, mOutput_size, mInput_size*mOutput_size);
        Vec b = new Vec(weights, 0, mOutput_size);
        
         
        // System.out.println("activation function done: " +activation);
        for(int r =0 ; r< mOutput_size; r++)           //edited here
        {
            double value = 0;
            for(int c = 0; c < mInput_size; c++)
            {
                value += x.get(c)*M.get(r*mInput_size + c);    
            }
            activation.set(r, value);
        } 
        activation.add(b);
    } 
  
    int total_weight_dimention(){
        int dimention = mOutput_size*mInput_size + mOutput_size;
        return dimention;
    }

/*    public void crossMult(Vec x, Matrix dummy) {
        for(int i = 0; i < size(); i++) {
            Vec temp = new Vec(0);
            temp.copy(x);
            temp.scale(get(i));
            dummy.takeRow(temp.vals);
        }
    }*/
    
    
    private static Matrix outer_product(Vec x, Vec y)
    {
        Matrix ret = new Matrix(0, y.size());

        for (int r= 0; r<x.size(); r++)
        {
            double [] products =new double[y.size()];
            for(int c=0; c<y.size(); c++)
            {
                products[c] = x.get(r)*y.get(c);
            }
            ret.takeRow(products);
        }
        return ret;
    }
    
    static void vecCrossMultiplyAdd(Vec x, Vec y, Matrix addTo)
    {       
        if( addTo.cols() != y.size() || addTo.rows() != x.size())
        {
            System.err.println("Size matching error: addTo.cols() != y.size() || addTo.rows() != x.size()");
            return;
        }
        
        for(int r = 0; r < x.size(); r++)            
        {
            Vec row = addTo.row(r);
            for(int c= 0; c< y.size(); c++)
            {
                double value = x.get(r)*y.get(c);
                row.set(c, value + row.get(c));
                
            }
        }    
    }
    

    
    private static Vec getColumnwiseMean(Matrix m)
    {
        Vec v = new Vec(m.cols());
        for (int i =0; i< m.cols(); i++)
        {
            v.set(i,m.columnMean(i));            
        }
        return v;   
    }
    public double get(int index)
    {
	return vals[start + index];
    }

    void ordinary_least_squares(Matrix X, Matrix Y, Vec weights)
    {
        Matrix y_yMean_cross_x_xMean = new Matrix(mOutput_size, mInput_size);
        Matrix x_xMean_cross_x_xMean = new Matrix(mInput_size, mInput_size);
        
        Vec xMean = getColumnwiseMean(X);
        Vec yMean = getColumnwiseMean(Y);
        
        for (int i=0; i<X.rows(); i++)
        {
            Vec row_x = new Vec(0);
            row_x.copy(X.row(i));
            Vec row_y = new Vec(0);
            row_y.copy(Y.row(i));
            
            // Calculate Means first, then subtract from the original value
            for(int c = 0; c<row_x.size(); c++)
            {
                row_x.set(c, row_x.get(c)- xMean.get(c));
            }
            
            for(int c=0; c<row_y.size(); c++)
            {
                row_y.set(c, row_y.get(c));
            }
            
            vecCrossMultiplyAdd(row_y, row_x, y_yMean_cross_x_xMean);
            vecCrossMultiplyAdd(row_x, row_x, x_xMean_cross_x_xMean);
        }
        
        x_xMean_cross_x_xMean = x_xMean_cross_x_xMean.pseudoInverse();
        
        Matrix M = Matrix.multiply(y_yMean_cross_x_xMean, x_xMean_cross_x_xMean, false, false);
        //System.out.println("M-->\n"+M.toString());
        
        Vec B = new Vec(mOutput_size);
        
        for(int i = 0; i< mOutput_size; i++)
        {
            Vec row = M.row(i);
            B.set(i, yMean.get(i) - row.dotProduct(xMean));
        }

        //storing B first in weights

        Vec bWeights = new Vec(weights, 0, B.size());
        bWeights.add(B);
        
        //Storing M after storing B in weights
        for(int i = 0; i<M.rows(); i++)
        {
            for(int j = 0; j < M.cols(); j++)
            {       
                weights.set(i*M.cols()+j+mOutput_size, M.row(i).get(j));
            }
        }
 
    } 
    
    
    static void OLSUnitTest() throws Exception
    {
        System.out.println("Executing Ordinary Least Squares unit test in LinearLayer...");
        Random rand = new Random();
        Vec weights = new Vec(3);
        

        Matrix Y = new Matrix(0,1);
        LayerLinear layer = new LayerLinear(2,1);
        
        Matrix X = new Matrix(0,2);
        double[] row = X.newRow();
        row[0] = rand.nextInt(5);
        row[1] = rand.nextInt(5);
        row = X.newRow();
        row[0] = rand.nextInt(5);
        row[1] = rand.nextInt(5);
        row = X.newRow();
        row[0] = rand.nextInt(5);
        row[1] = rand.nextInt(5);
        
        weights.set(0, rand.nextInt(5));
        weights.set(1, rand.nextInt(5));
        weights.set(2, rand.nextInt(5));


        
        for(int i = 0; i< X.rows(); i++) {
            layer.activate(weights, X.row(i));
            row = Y.newRow();
            row[0] = layer.activation.get(0)+ rand.nextInt(2);
            System.out.println("\n Activate function            =\n"+ row[i]);            
        }
        
        Vec weightsOLS = new Vec(3);
        layer.ordinary_least_squares(X, Y, weightsOLS);
        
        double SumSqrErr = weightsOLS.squaredDistance(weights);
        
        System.out.println("\nRandom X            =\n" +  X.toString());
        System.out.println("\nRandomly choosen Weights(b, m)   = " +  weights);
        System.out.println("Predicted Weights    = " + weightsOLS.toString()+"\n");

        if(SumSqrErr>15)
            throw new Exception("Error!! Thhe Sum Squared Error is greater than 15, error is " + SumSqrErr);
        else
            System.out.println("LinearLayer::Ordinary Least Squares unit test:: root-mean-squared-error = " + SumSqrErr);

      //  System.out.println("");
    }
        

    @Override
    void backProp(Vec weights, Vec prevBlame) {
        
       // System.out.println("Traing backprop\n");
        Matrix M = new Matrix(0, mInput_size);
        for(int j=0; j<mOutput_size; j++) {
            Vec M_row = new Vec(mInput_size);
            for(int i=0; i<M_row.len; i++) {
                M_row.set(i, weights.get(j*mInput_size + mOutput_size + i));                
            }
                M.takeRow(M_row.vals);
        }       
        Matrix M_transpose = M.transpose();
        
        for(int i=0; i< M_transpose.rows(); i++) {       
            Vec row = M_transpose.row(i);            
            prevBlame.set(i, row.dotProduct(blame));   
        }        
   
    }
 
    
    
   
    @Override
    void updateGradient(Vec x, Vec gradient) {
        Vec b = new Vec(mOutput_size);
        for(int i=0; i<b.len; i++) {
            //double k = blame.get(i+mOutput_size*mInput_size);
            //b.set(i+mOutput_size*mInput_size, k);
            double k = blame.get(i); //modified 
            b.set(i, k);
        }
        Matrix gm = new Matrix(0, mInput_size); //need to check        
        blame.crossMult(x, gm);
	for(int i = 0; i < mOutput_size; i++)
            gradient.set(i, b.get(i));
		
        for(int j = 0; j <gm.rows(); j++) {            
            for(int k = 0; k <gm.cols(); k++)
                gradient.set(k+j*gm.cols()+mOutput_size, gm.row(j).get(k));
        }        
    }
    
    
    	public void activationUnitTest()
	{
		System.out.println("Performing LinearLayer::Activation unit test...");
		double[] d_x = {0, 1, 2};
		double[] d_y = {9,6};
		double[] d_bm = {1, 5, 1, 2, 3, 2, 1, 0};
		
		Vec x = new Vec(d_x);
		Vec y = new Vec(d_y);
		Vec weights = new Vec(d_bm);
		
		LayerLinear layer = new LayerLinear(3, 2);
		layer.activate(weights, x);
		System.out.println("X            = "+ x.toString());
		System.out.println("Weights(b,m) = "+ weights.toString());
		System.out.println("Actual Y     = "+ y.toString());
		System.out.println("Predicted Y  = "+ layer.activation.toString()+ "\n");
		System.out.println("\nLinearLayer::Activation Unit test successfully done\n");
		
	}

}