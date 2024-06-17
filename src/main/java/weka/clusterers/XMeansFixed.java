/**
 * 
 */
package weka.clusterers;

import java.util.Random;

import weka.core.Instances;
import weka.tools.InstancesTools;

/**
 * Fixed Xmeans
 * @author pawel trajdos
 * @since 0.0.5
 * @version 0.0.5
 */
public class XMeansFixed extends XMeans {

	private static final long serialVersionUID = 8623299605516980769L;
	/**
	 * Contains the number of unique instances
	 */
	protected int m_UniqueInstancesNumber = 0;

	

	@Override
	public void buildClusterer(Instances data) throws Exception {
		this.m_UniqueInstancesNumber = InstancesTools.countUniqieInstances(data);
		super.buildClusterer(data);
	}


	@Override
	protected Instances makeCentersRandomly(Random random0, Instances model, int numClusters) {
		int numEffectiveClusters = Math.min(this.m_UniqueInstancesNumber, numClusters);
		
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
