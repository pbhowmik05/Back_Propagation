import java.util.Random;
import java.util.ArrayList;



class Main
{
    static void backPropTest(SupervisedLearner learning) {
        //Matrix data = new Matrix();
        //data.loadARFF("data/hypothyroid.arff");
	Matrix trainingFeatures = new Matrix();
        Matrix trainingLabels = new Matrix();
        //Matrix validationFeatures = new Matrix();
        //Matrix validationLabels = new Matrix();
        Matrix testFeatures = new Matrix();
        Matrix testLabels = new Matrix();
        Random random = new Random();
        
        //int nCount = data.valueCount(0);
        //System.out.println("ncount: "+nCount);  
        
        trainingFeatures.loadARFF("data/train_feat.arff");
        trainingLabels.loadARFF("data/train_lab.arff");
        testFeatures.loadARFF("data/test_feat.arff");
        testLabels.loadARFF("data/test_lab.arff");
        
        //Feature data scaling down
        double scale_factor = 1/256.0;
        trainingFeatures.scale(scale_factor);
        testFeatures.scale(scale_factor);
        
        learning.train(trainingFeatures, trainingLabels);
        
        
        int rowCountFeat = trainingFeatures.rows();
        
        //ArrayList<Integer> randLayer = new ArrayList<>();

        //int nDataSetCount = trainfeatures.rows();

        
	System.out.println("Traing done\n");
	double learning_rate = 0.03;
        int vec_size= 10;
        
	for (int i = 0; i < 9; i++) {
		
            ArrayList<Integer> randLayer = new ArrayList<Integer>();
            int rnd = randLayer.size();
          //  for (int m = 0; randLayer.size() < rowCountFeat; m++) {
            while(randLayer.size() < rowCountFeat) {    
		int pointer = random.nextInt(rowCountFeat);
                boolean retval = randLayer.contains(pointer); 
		if (!retval==true) {
                    randLayer.add(pointer);
		}
            }
            System.out.println("Randomization Completed\n");	
                
            for (int j = 0; j < trainingFeatures.rows(); j++) {
		Vec trainFeaturesVec = trainingFeatures.row(randLayer.get(j));
		Vec LabelVec = new Vec(vec_size);
		LabelVec.fill(0);
		LabelVec.set((int) trainingLabels.row(randLayer.get(j)).get(0), 1);
            //    System.out.println("test start\n");
                ((NeuralNet) learning).refineWeights(trainFeaturesVec, LabelVec, learning_rate);
            }
            //int nCount = learning.countMisclassifications(testFeatures, testLabels);
			//System.out.println("Epoch " + (i + 1)+" : Misclassification " + nCount + " out of " + testLabels.rows() * testLabels.cols());
			
            int Count_misclassification = learning.countMisclassifications(testFeatures, testLabels);
            System.out.println(+(i+1)+" Iteration " + i +" : Calculated Misclassification rate = " + Count_misclassification + " / " + testLabels.rows() * testLabels.cols());
			
        }
        
        
    }

    public static void modeltrain(SupervisedLearner learning) {
	//System.out.println("Traing testlearner\n");	
            backPropTest(learning);
        //LayerLinear layer = new LayerLinear(2, 1);
         //   try {
           // layer.OLSUnitTest();
            //System.out.println("Traing testlearner\n");
       // } catch (Exception ex) {
            //Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
       // }
	}

	public static void main(String[] args)
	{
		modeltrain(new NeuralNet());
                //System.out.println("Traing main\n");
	}
}