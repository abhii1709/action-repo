function getTime(){
	let today = new Date();
	document.getElementById('date').innerHTML = today.toString().slice(0, 24);
	setInterval("getTime()", 1000);
}
getTime();
document.querySelector('#year').innerHTML = new Date().getFullYear();
