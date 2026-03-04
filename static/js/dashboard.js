

fetch("http://127.0.0.1:5000/data")
.then(res => res.json())
.then(dataset => {

    // -------- KPI Calculations --------
    let total = dataset.length;
    let churnCount = dataset.filter(d => d.Churn == 1).length;
    let churnRate = ((churnCount/total)*100).toFixed(2);

    document.getElementById("totalCustomers").innerText = total;
    document.getElementById("churnRate").innerText = churnRate + "%";

    // -------- Prepare Data --------
    dataset.forEach(d => {
        d.status = d.Churn == 1 ? "Churn" : "Stay";
        d.Age = +d.Age;
    });

    var cf = crossfilter(dataset);

    // Churn Distribution
    var statusDim = cf.dimension(d => d.status);
    var statusGrp = statusDim.group();

    new dc.PieChart("#pieChart")
        .dimension(statusDim)
        .group(statusGrp);

    // Age vs Churn
    var ageDim = cf.dimension(d => Math.floor(d.Age/10)*10);
    var ageGrp = ageDim.group();

    new dc.BarChart("#barChart")
        .dimension(ageDim)
        .group(ageGrp)
        .x(d3.scaleBand())
        .xUnits(dc.units.ordinal);

    dc.renderAll();
});


// -------------------------------
// Fetch Metrics
// -------------------------------
fetch("http://127.0.0.1:5000/metrics")
.then(res => res.json())
.then(data => {

    // Accuracy
    document.getElementById("accuracy").innerText = data.accuracy;

    // Confusion Matrix
    let cm = data.confusion_matrix;

    document.getElementById("confusionMatrix").innerHTML =
        "<table class='table table-bordered text-center'>" +
        "<tr><th></th><th>Pred Stay</th><th>Pred Churn</th></tr>" +
        "<tr><th>Actual Stay</th><td>"+cm[0][0]+"</td><td>"+cm[0][1]+"</td></tr>" +
        "<tr><th>Actual Churn</th><td>"+cm[1][0]+"</td><td>"+cm[1][1]+"</td></tr>" +
        "</table>";
});




// -------------------------------
// Fetch Feature Importance
// -------------------------------
fetch("http://127.0.0.1:5000/feature-importance")
.then(res => res.json())
.then(data => {

    data.forEach(d => {
        d.importance = +d.importance;
    });

    var cfFI = crossfilter(data);
    var featureDim = cfFI.dimension(d => d.feature);
    var featureGrp = featureDim.group().reduceSum(d => d.importance);

    new dc.BarChart("#featureChart")
        .dimension(featureDim)
        .group(featureGrp)
        .x(d3.scaleBand())
        .xUnits(dc.units.ordinal);

    dc.renderAll();
});




// -------------------------------
// 3️⃣ Predict Function
// -------------------------------
function predict(){
    fetch("http://127.0.0.1:5000/predict",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
            Age:parseFloat(age.value),
            Gender:parseFloat(gender.value),
            MonthlySpend:parseFloat(spend.value),
            Tenure:parseFloat(tenure.value),
            VisitsPerMonth:parseFloat(visits.value)
        })
    })
    .then(res=>res.json())
    .then(data=>{
        document.getElementById("predictionResult").innerHTML =
            "Prediction: " + (data.prediction==1?"CHURN":"STAY") +
            "<br>Probability: " + data.probability +
            "<br>Risk: " + data.risk_level;
    });
}

