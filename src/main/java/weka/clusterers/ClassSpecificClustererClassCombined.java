/**
 * 
 */
package weka.clusterers;

import weka.core.Instance;
import weka.core.Utils;

/**
 * @author pawel trajdos
 * @since 0.0.1
 * @version 0.0.1
 *
 */
public class ClassSpecificClustererClassCombined extends ClassSpecificClusterer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3221556640429200690L;

	/**
	 * 
	 */
	public ClassSpecificClustererClassCombined() {
		super();
	}
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		if(this.noInstances)
			return new double[] {1.0};
		if(this.noClass)
			return this.m_Clusterer.distributionForInstance(instance);
		
		double[] distribution = new double[this.numberOfClusters];
		
		this.removeFilter.input(instance);
		Instance filteredInstance = this.removeFilter.output();
		this.removeFilter.batchFinished();
		
		int[] maxRespIndices = new int[this.clusterers.length];
		double[] tmpResponse;
		for(int i=0;i<this.clusterers.length;i++) {
			tmpResponse = this.clusterers[i].distributionForInstance(filteredInstance);
			maxRespIndices[i] = Utils.maxIndex(tmpResponse);
		}
		
		distribution[this.getClusterIndex(maxRespIndices)] =1.0;
		
		
		return distribution;
	}
	
	protected int getClusterIndex(int[] maxRespIndices) throws Exception {
		int cumulatedBase=1;
		int clusterIndex=0;
		int tmpNumberOfClusters=0;
		for(int i=0;i<this.clusterers.length;i++) {
			tmpNumberOfClusters = this.clusterers[i].numberOfClusters();
			if(tmpNumberOfClusters == 1) 
				continue;
		
			clusterIndex+= maxRespIndices[i]*cumulatedBase;
			
			cumulatedBase*= tmpNumberOfClusters;
			
		}
		return clusterIndex;
	}
	
	@Override
	protected void calculateNumberOfClusters() throws Exception {
		this.numberOfClusters=1;
		for(int i=0;i<this.clusterers.length;i++)
			this.numberOfClusters*= this.clusterers[i].numberOfClusters();
	}

}
