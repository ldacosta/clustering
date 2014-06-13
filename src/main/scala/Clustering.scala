import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeansModel, KMeans}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ Vector => mllibVector }
import org.apache.spark.mllib.linalg.{ Vectors => mllibVectors }

/**
 * Created by LDacost1 on 2014-06-10.
 */
object Clustering {
  def main(args: Array[String]) {
    val dataFile = "C:\\Users\\ldacost1\\data\\phoneplans.txt"
    val conf = new SparkConf().setAppName("Clustering Application")
    val sc = new SparkContext(conf)
    val parsedData = readDataFromFile(sc, dataFile)
    findBestClusterSize(parsedData)
  }

  def readDataFromFile(sc: SparkContext, fileName: String): RDD[mllibVector] = {
    val data = sc.textFile(fileName) // ("data/kmeans_data.txt")
    data.map(s => mllibVectors.dense(s.split('\t').map(_.toDouble)))
  }

  def normalizeData(data: RDD[mllibVector]): RDD[mllibVector] = {
    def fMin(v1: mllibVector, v2: mllibVector): mllibVector = {
      val xx = v1.toArray zip v2.toArray
      val asArray = xx.foldLeft(Array[Double]()) { case (a, (val1, val2)) => a :+ Math.min(val1, val2)}
      mllibVectors.dense(asArray)
    }

    def fMax(v1: mllibVector, v2: mllibVector): mllibVector = {
      val xx = v1.toArray zip v2.toArray
      val asArray = xx.foldLeft(Array[Double]()) { case (a, (val1, val2)) => a :+ Math.max(val1, val2)}
      mllibVectors.dense(asArray)
    }

    if (data.count() == 0)
      data
    else {
      // min value of each dimension must be ZERO
      val minValues = data.fold(data.first)(fMin)
      val withMinZero = data.map(v => mllibVectors.dense({ (v.toArray zip minValues.toArray).map{ case (aVal, m) => aVal - m } }))
      val maxValues = withMinZero.fold(withMinZero.first)(fMax)
      val withMaxOne = withMinZero.map(v => mllibVectors.dense({ (v.toArray zip maxValues.toArray).map{ case (aVal, m) => aVal / m } }))
      withMaxOne
    }
  }

  def findBestClusterSize(parsedData: RDD[mllibVector]): KMeansModel = {
    def showVector(v: mllibVector) = {
      println(s"{${v.toArray.mkString(",")}}")
    }
    val possibleNumClusters: List[Int] = (1 to (parsedData.count().asInstanceOf[Int])).toList
    val numIterations = 20
    val allOptions =
      possibleNumClusters.foldLeft(List[(KMeansModel, Double)]()) { (results, numClusters) =>
        val clusters = KMeans.train(parsedData, numClusters, numIterations, runs = 10)
        val WSSSE = clusters.computeCost(parsedData)
        // println(s"k = $numClusters, centroids ==> ");
        // clusters.clusterCenters.foreach { v => println(s"{${v.toArray.mkString(",")}}")};
        // println(s"k = $numClusters, WSSSE = $WSSSE")
        results ++ List((clusters, WSSSE))
      }.toArray
    val allDistances = allOptions.map(_._2)
    val firstDerivative = (allDistances.take(allDistances.length - 1) zip allDistances.tail).map{ case (d1, d2) => d2 - d1}
    val secondDerivative = (firstDerivative.take(firstDerivative.length - 1) zip firstDerivative.tail).map{ case (d1, d2) => d2 - d1}

    // look for place in second derivative that is in a "sink":
    val distanceIndex =
      (secondDerivative.zipWithIndex.collectFirst { case (aVal, theIndex)
        if ((theIndex > 0) && (secondDerivative(theIndex - 1) > aVal) && (secondDerivative(theIndex + 1) > aVal)) => theIndex } match {
        case None => secondDerivative.length - 1
        case Some(theIndex) => theIndex
      }) + 2 // + 2 because I go from 2nd derivative to actual values

    val allModels = allOptions.map(_._1)
    val bestModel = allModels(distanceIndex)
    println(s"I chose model #$distanceIndex(out of (${secondDerivative.length - 1})  (k = ${bestModel.k})")

    parsedData.foreach{ v => showVector(v); println(s"Element in cluster ${ bestModel.predict(v)}")}

    bestModel
  }

}
