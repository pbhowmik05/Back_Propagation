// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

class BaselineLearner extends SupervisedLearner
{
	double[] mode;

        @Override
	String name()
	{
		return "Baseline";
	}

        @Override
	void train(Matrix features, Matrix labels)
	{
		mode = new double[labels.cols()];
		for(int i = 0; i < labels.cols(); i++)
		{
			if(labels.valueCount(i) == 0)
				mode[i] = labels.columnMean(i);
			else
				mode[i] = labels.mostCommonValue(i);
		}
	}

        @Override
	Vec predict(Vec in)
	{
		return new Vec(mode);
	}
 //       @Override
   //     Vec backProp()
}
