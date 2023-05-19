/**
 * 
 */
package weka.clusterers;

import java.util.Arrays;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.tools.GlobalInfoHandler;
import weka.tools.data.InstancesOperator;

/**
 * A class for clusterers that make separate clustering for each available class
 * @author pawel trajdos
 * @since 0.0.1
 * @version 0.0.1
 * 
 *
 */
public class ClassSpecificClusterer extends SingleClustererEnhancer implements GlobalInfoHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7492315438320617468L;
	
	protected Remove removeFilter;
	
	protected boolean noClass=false;
	
	protected boolean noInstances=false;

	protected boolean classesOnly = false;
	
	protected Clusterer[] clusterers;
	
	protected int numberOfClusters;

	protected int numberOfClasses;

	/**
	 * 
	 */
	public ClassSpecificClusterer() {
		super(); 
	}

	@Override
	public void buildClusterer(Instances data) throws Exception {
		this.noClass=false;
		this.noInstances = false;
		int classIndex = data.classIndex();
		int numInstances = data.numInstances();
		
		if(classIndex <0) {
			this.noClass = true;
			this.numberOfClasses = 0; 
			if(!this.m_DoNotCheckCapabilities)
				this.m_Clusterer.getCapabilities().testWithFail(data);
			this.m_Clusterer.buildClusterer(data);
			this.numberOfClusters = this.m_Clusterer.numberOfClusters();
			return;
		}

		
		if(numInstances == 0) {
			this.noInstances=true;
			this.numberOfClasses = data.numClasses();
			this.numberOfClusters= this.numberOfClasses;
			return;
		}

		int nAttributes = data.numAttributes();
		if (classIndex>=0 & nAttributes==1){
			this.classesOnly = true;
			this.numberOfClasses = data.numClasses();
			this.numberOfClusters = this.numberOfClasses;
			return;
		}
		
		Instances[] classSplitData = InstancesOperator.classSpecSplit(data);
		this.numberOfClasses = classSplitData.length;
		int numClasses = classSplitData.length;
		this.clusterers = AbstractClusterer.makeCopies(this.m_Clusterer, numClasses);
		
		Remove remFilter  = new Remove();
		remFilter.setAttributeIndicesArray(new int[] {classIndex});
		remFilter.setInputFormat(data);
		remFilter.setInvertSelection(false);
		this.removeFilter = remFilter;
		
		Instances tmpInstances = null;
		for(int i=0;i<numClasses;i++) {
			tmpInstances = Filter.useFilter(classSplitData[i], this.removeFilter);
			if(!this.m_DoNotCheckCapabilities)
				this.clusterers[i].getCapabilities().testWithFail(tmpInstances);
			this.clusterers[i].buildClusterer(tmpInstances);
		}
		
		this.calculateNumberOfClusters();
		
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		
		if(this.noClass)
			return this.m_Clusterer.distributionForInstance(instance);

		if(this.classesOnly | this.noInstances){
			double[] result = new double[this.numberOfClasses];
			Arrays.fill(result, 0, result.length, 1);
			Utils.normalize(result);
			return result;
		}
		
		this.removeFilter.input(instance);
		Instance filteredInstance = this.removeFilter.output();
		this.removeFilter.batchFinished();
		int numOfClusters = this.numberOfClusters();
		double[] distribution = new double[numOfClusters];
		
		int cnt=0;
		for(int i=0;i<this.clusterers.length;i++) {
			double[] tmpDist = this.clusterers[i].distributionForInstance(filteredInstance);
			for(int j=0;j<tmpDist.length;j++)
				distribution[cnt++]=tmpDist[j];
		}
			
		double sum = Utils.sum(distribution);
		if(!Utils.eq(sum, 0))
			Utils.normalize(distribution);
		
		
		return distribution;
	}
	
	@Override
	public int numberOfClusters() throws Exception {
		return this.numberOfClusters;
	}

	/**
	 * Returns the number of classes
	 * @return 0 if no classes during training or value greater or equal one.
	 * @throws Exception
	 */
	public int numberOfClasses()throws Exception{

		return this.numberOfClasses;

	}

	/**
	 * Returns class-specific cluster responses
	 * @param instance
	 * @return array of class-specific cluster responses
	 * @throws Exception
	 */
	public double[][] classSpecificDistributionForInstance(Instance instance) throws Exception{

		if(this.noClass){
			return  new double[][] {this.distributionForInstance(instance)};
		}

		if(this.classesOnly | this.noInstances){
			double[][] result = new double[this.numberOfClasses][1];

			for(int i=0;i<result.length;i++){
				result[i][0] = 1.0;
			}

			return result;
		}

		double [][] distribution = new double[this.clusterers.length][];

		this.removeFilter.input(instance);
		Instance filteredInstance = this.removeFilter.output();
		this.removeFilter.batchFinished();

		for(int i =0; i<this.clusterers.length;i++){
			distribution[i] = this.clusterers[i].distributionForInstance(filteredInstance);
		}

		return distribution;
	}

	public double[][][] classSpecificDistrbutionForInstances(Instances instances) throws Exception{
		
		int numInstances = instances.numInstances();

		double[][][] distribution = new double[numInstances][][];

		for(int i=0;i<numInstances;i++){
			Instance instance = instances.get(i);
			distribution[i] = this.classSpecificDistributionForInstance(instance);
		}

		return distribution;
	}

	public int[] numberOfClassSpecificClusters() throws Exception{

		if(this.noClass){
			return new int[] {this.m_Clusterer.numberOfClusters()};
		}

		if(this.classesOnly | this.noInstances){
			int[] result = new int[this.numberOfClasses];
			Arrays.fill(result, 0, this.numberOfClasses , 1);
			return result;
		}

		int[] clustersNumber = new int[this.clusterers.length];

		for(int c=0; c<this.clusterers.length; c++){
			clustersNumber[c] = this.clusterers[c].numberOfClusters();
		}
		return clustersNumber;
	}
	
	protected void calculateNumberOfClusters() throws Exception {
		this.numberOfClusters=0;
		for(int i=0;i<this.clusterers.length;i++)
			this.numberOfClusters+= this.clusterers[i].numberOfClusters();
	}

	@Override
	public String globalInfo() {
		return "Creates homogeneous clusters containing objects from one class";
	}
	
	@Override
	public Capabilities getCapabilities() {
		Capabilities caps = super.getCapabilities();
		caps.disable(Capability.NO_CLASS);
		
		caps.enable(Capability.NOMINAL_CLASS);
		caps.enable(Capability.STRING_CLASS);
		
		 // set dependencies
	    for (Capability cap : Capability.values()) {
	      caps.enableDependency(cap);
	    }
		
		return caps;
	}

}
