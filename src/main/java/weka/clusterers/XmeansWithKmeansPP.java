/**
 * 
 */
package weka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.util.Pair;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.UtilsPT;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.tools.InstancesTools;

/**
 * Class implements Xmeans with the initialization using K-Means++
 * @author pawel trajdos
 * @since 0.0.4
 * @version 0.0.6
 *
 */
public class XmeansWithKmeansPP extends XMeans {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6645825577412294656L;
	
	/**
	 * Contains the number of unique instances
	 */
	protected int m_UniqueInstancesNumber = 0;
	public static int nRepeats = 100;

	/**
	 * 
	 */
	public XmeansWithKmeansPP() {
		super();
	}
	
	
	
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
	    this.m_NumClusters = numEffectiveClusters;
	    
	    
	    
	    if(numEffectiveClusters == this.m_UniqueInstancesNumber) {
	    	for(Instance instance: m_Instances) {
	    		clusterCenters.add(instance);
	    	}
	    	return clusterCenters;
	    }
	    
	    
	    Map<Integer,Integer> instanceMap = new HashMap<>();
	    
	    int instIndex = Math.abs(random0.nextInt()) % this.m_Instances.numInstances();
	    Instance lastSelected = this.m_Instances.get(instIndex);
	    clusterCenters.add(lastSelected);
	    
	    instanceMap.put(Arrays.hashCode(lastSelected.toDoubleArray()),1);
	    
	    Instance tmpInstance = null;
	    
	    for(int i=1;i<this.m_NumClusters;i++) {
	    	int counter = 0;
	    	while(true) {
	    		counter++;
	    		
	    		if (counter > XmeansWithKmeansPP.nRepeats) {
	    			//Safety break
	    			this.m_NumClusters = clusterCenters.size();
	    			return clusterCenters;
	    		}
	    		
	    		tmpInstance = this.selectNextCenter(lastSelected, random0);
	    		if (instanceMap.containsKey( Arrays.hashCode(tmpInstance.toDoubleArray()) ))
	    			continue;
	    		
	    		clusterCenters.add(tmpInstance);
		    	lastSelected = tmpInstance;
		    	instanceMap.put(Arrays.hashCode(tmpInstance.toDoubleArray()), 1);
		    	break;
	    		
	    	}
	    	
	    	
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
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		
		this.m_ReplaceMissingFilter.input(instance);
	    Instance inst = this.m_ReplaceMissingFilter.output();
	    
	    double[] dists = new double[this.numberOfClusters()];
	    for(int i =0;i<dists.length;i++) {
	    	dists[i] = this.m_DistanceF.distance(inst, this.m_ClusterCenters.instance(i));
	    }
	    double[] distribution = UtilsPT.softMin(dists);
	    
		return distribution;
	}
	
	

}
