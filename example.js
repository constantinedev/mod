<script>
  var tag_veiew = document.getElementByTagName('tag_name');
  //Change the element
  element.innerHTML = 'HTML'
  element.attrbite = 'value'
  element.style.property = 'style'

  var add_element = document.createElemenet(element)
  add_element.onclick = function() {
    pass
  };

  /// JS DOM GET REQUEST
  fetch('url)
    .then(response => response.json())
    .then(data => console.log(data))
  
  /// JS DOM POST REQUEST
  fetch('url', {
    mehtod: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
      .then(repsponse => response.json())
      .then(data => console.log(data));
  }

  document.getElementByTagName('tag_name').onclick = fx_fucntion()
  
  /// FUNCTION
  function fx_example(){
    console.log('123');
  }
</script>
