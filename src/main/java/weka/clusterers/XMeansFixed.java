/**
 * 
 */
package weka.clusterers;

import java.util.Random;

import weka.core.Instances;

/**
 * 
 */
public class XMeansFixed extends XMeans {

	private static final long serialVersionUID = 8623299605516980769L;

	/**
	 * 
	 */
	public XMeansFixed() {
		// TODO Auto-generated constructor stub
	}

	@Override
	protected Instances makeCentersRandomly(Random random0, Instances model, int numClusters) {
		int numTrainingInstances = this.m_Instances.numInstances();
		int numEffectiveClusters = Math.min(numTrainingInstances, numClusters);
		
		 Instances clusterCenters = new Instances(model, numEffectiveClusters);
		    m_NumClusters = numEffectiveClusters;

		    // makes the new centers randomly
		    for (int i = 0; i < numEffectiveClusters; i++) {
		      int instIndex = Math.abs(random0.nextInt()) % m_Instances.numInstances();
		      clusterCenters.add(m_Instances.instance(instIndex));
		    }
		    return clusterCenters;
	}
	
	

}
