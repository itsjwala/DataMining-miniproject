$(document).ready(()=>{

function generateOutput(value){
	if(value===true)
	return `
	<div class="alert alert-success alert-dismissible">
	  <button type="button" class="close" data-dismiss="alert">&times;</button>
	    Customer should be shown the Ads
	</div>
	`

	else{
	return `
	<div class="alert alert-success alert-dismissible">
	  <button type="button" class="close" data-dismiss="alert">&times;</button>
	    Customer should not be shown the Ads
	</div>
	`
	}
}

$('#submit').click((e)=>{
	e.preventDefault();
	var age=$('#age').val()
	var salary=$('#salary').val()
	var algo=$('#algo').val()
	$.getJSON(`/predict?age=${age}&salary=${salary}&algo=${algo}`, function(result){

		$("#output").html(generateOutput(result.prediction))
    });

})

})