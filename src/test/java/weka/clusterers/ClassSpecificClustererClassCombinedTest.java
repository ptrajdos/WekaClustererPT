package weka.clusterers;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.tools.data.RandomDataGenerator;

public class ClassSpecificClustererClassCombinedTest extends ClassSpecificClustererTest {

	public ClassSpecificClustererClassCombinedTest(String name) {
		super(name);
	}
	
	@Override
	public Clusterer getClusterer() {
		return new ClassSpecificClustererClassCombined();
	}
	
	public void testClusterCoverage() {
		RandomDataGenerator gen = new RandomDataGenerator();
		gen.setNumClasses(5);
		
		Instances data = gen.generateData();
		
		Clusterer clust = this.getClusterer();
		
		try {
			clust.buildClusterer(data);
			int numClusters = clust.numberOfClusters();
			double[] coverage = new double[numClusters];
			for (Instance instance : data) {
				this.addToArray(coverage, clust.distributionForInstance(instance));
			}
			
			double sum = Utils.sum(coverage);
			assertTrue("Coverage sum", Utils.eq(sum, data.numInstances()) );
			
		} catch (Exception e) {
			fail("An exception has been caught: " + e.getMessage());
		}
		
	}
	
	protected void addToArray(double[] accum, double[] arg) throws Exception {
		if(accum.length != arg.length)
			throw new Exception("Incompatible lengths");
		
		for(int i=0;i<accum.length;i++)
			accum[i]+=arg[i];
	}


}
