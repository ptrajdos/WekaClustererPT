/**
 * 
 */
package weka.clusterers;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.tools.InstancesTools;

/**
 * Fixed Xmeans
 * @author pawel trajdos
 * @since 0.0.4
 * @version 0.0.6
 */
public class XMeansFixed extends XMeans {

	private static final long serialVersionUID = 8623299605516980769L;
	/**
	 * Contains the number of unique instances
	 */
	protected int m_UniqueInstancesNumber = 0;

	

	@Override
	public void buildClusterer(Instances data) throws Exception {
		
		RemoveDuplicates removeDuplicates = new RemoveDuplicates();
		removeDuplicates.setInputFormat(data);
		Instances noDupsData = Filter.useFilter(data, removeDuplicates);
		
		this.m_UniqueInstancesNumber = InstancesTools.countUniqieInstances(noDupsData);
		super.buildClusterer(noDupsData);
	}


	@Override
	protected Instances makeCentersRandomly(Random random0, Instances model, int numClusters) {
		int numEffectiveClusters = Math.min(this.m_UniqueInstancesNumber, numClusters);
		
		 Instances clusterCenters = new Instances(model, numEffectiveClusters);
		    m_NumClusters = numEffectiveClusters;
		    
		    if(numEffectiveClusters == this.m_UniqueInstancesNumber) {
		    	for(Instance instance: m_Instances) {
		    		clusterCenters.add(instance);
		    	}
		    	return clusterCenters;
		    }
		    //At this moment it is sure that the number of unique instances is greater than the number of clusters

		    // makes the new centers randomly
		    //All instances in the data are unique
		    //Map assures unique centers
		    Map<Integer,Integer> uniqCentersMap = new HashMap<>();
		    
		    for (int i = 0; i < numEffectiveClusters; i++) {
		    	while(true) {
			    	int instIndex = Math.abs(random0.nextInt()) % m_Instances.numInstances();
			    	
			    	if( uniqCentersMap.containsKey(instIndex) )
			    		continue;
			    	
			    	uniqCentersMap.put(instIndex, 1);
			    	clusterCenters.add(m_Instances.instance(instIndex));
			    	break;
		    	}
		      
		      
		    }
		    return clusterCenters;
	}
	
	

}
