document.getElementById("upload-photo").onchange = (e) => {
    var filename = e.target.files[0].name;
    var allowedExtensions =  
                    /(\.jpg|\.jpeg|\.png)$/i; 
              
    if (!allowedExtensions.exec(filename)) { 
        alert('Invalid file type: Please provide an img'); 
        fileInput.value = ''; 
        return false; 
    }  
    if(filename.length > 10)
        filename =filename.substr(0, 7) + '...'
    document.getElementById("upload-label").innerHTML = `<span class='orange'>${filename}</span> selected`;
    if(e.target.files.length != 0)
    {
        document.getElementById('sub-btn').style.display = 'block'
        window.scrollTo(0,document.body.scrollHeight);
    }
};