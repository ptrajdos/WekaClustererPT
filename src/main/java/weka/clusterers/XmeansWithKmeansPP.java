/**
 * 
 */
package weka.clusterers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.util.Pair;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class implements Xmeans with the initialization using K-Means++
 * @author pawel trajdos
 * @since 0.0.1
 * @version 0.0.1
 *
 */
public class XmeansWithKmeansPP extends XMeans {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6645825577412294656L;

	/**
	 * 
	 */
	public XmeansWithKmeansPP() {
		super();
	}
	
	@Override
	protected Instances makeCentersRandomly(Random random0, Instances model, int numClusters) {
		//return super.makeCentersRandomly(random0, model, numClusters);
		
		Instances clusterCenters = new Instances(model, numClusters);
	    this.m_NumClusters = numClusters;
	    
	    int instIndex = Math.abs(random0.nextInt()) % this.m_Instances.numInstances();
	    Instance lastSelected = this.m_Instances.get(instIndex);
	    clusterCenters.add(lastSelected);
	    
	    if(this.m_NumClusters==1)
	    	return clusterCenters;
	    
	    Instance tmpInstance = null;
	    for(int i=1;i<this.m_NumClusters;i++) {
	    	tmpInstance = this.selectNextCenter(lastSelected, random0);
	    	clusterCenters.add(tmpInstance);
	    	lastSelected = tmpInstance;
	    }
	    
		
		return clusterCenters;
		
	}
	
	protected List<Pair<Instance,Double>> getSquaredDistances(Instance srcInst){
		int numInst  = this.m_Instances.numInstances();
		double dist=0;
		double[] dists = new double[numInst];
		double sum=0;
		for(int i=0;i<numInst;i++) {
			dist = this.m_DistanceF.distance(this.m_Instances.get(i), srcInst);
			dist*=dist;
			dists[i]=dist;
			sum+=dist;
		}
		if(sum>0)
			Utils.normalize(dists);
		
		List<Pair<Instance,Double>> distList = new ArrayList<Pair<Instance, Double>>(numInst);
		for(int i=0;i<dists.length;i++) {
			distList.add(new Pair<Instance, Double>(this.m_Instances.get(i), dists[i]));
		}
		
		Collections.sort(distList, new Comparator<Pair<Instance,Double>>(){

			@Override
			public int compare(Pair<Instance, Double> o1, Pair<Instance, Double> o2) {
				double val1 = o1.getValue();
				double val2 = o2.getValue();
				
				if(val1<val2)
					return 1;
				if(val1>val2)
					return -1;
				
				return 0;
			}
			
		});
		return distList;
	}
	
	protected Instance selectNextCenter(Instance lastCenter, Random random) {
		List<Pair<Instance,Double>> sqList = this.getSquaredDistances(lastCenter);
		Instance selectedInstance = this.rouletteSelector(sqList, random);
		return selectedInstance;
	}
	
	protected Instance rouletteSelector(List<Pair<Instance,Double>> squaredDistList,Random random) {
		Instance selectedInstance=null;
		
		double rndVal = random.nextDouble();
		double sum=0;
		int cnt=0;
		int listLen = squaredDistList.size();
		while(sum<rndVal && cnt<listLen) {
			selectedInstance = squaredDistList.get(cnt).getKey();
			sum += squaredDistList.get(cnt).getValue();
			cnt++;
		}	
		return selectedInstance;
	}
	
	

}
