async function main() {
    async function sfetch(endpoint){
        let res = await fetch(endpoint)
        return res.json()
    }

    document.getElementById("test").addEventListener("click", async () => {
        let data = [await sfetch("test")]
        let bg_color = "antiquewhite"
        let layout = {
                title: "bardo est sus",
                paper_bgcolor: bg_color,
                plot_bgcolor: bg_color,
                xaxis: {
                    type: "date"
                }   
            }

        Plotly.newPlot("plot1", data, layout)
        
    })

}



main()