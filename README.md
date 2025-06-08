# Dizertatie

- Route name: `/braille`
- Input type: file upload
- Response type: JSON

1. Run the following snippet:
   `pip install -r requirements_braille.txt`

2. go to detection -> main.py -> run it.

## Notes

Response after request:

```JSON
{
    "text" : "YOUR TEXT HERE",
    "img_base64": [IMAGE_RESULT]
}
```

To process the result, maybe this example is useful:

```Typescript
 const handleUpload = async () => {
    if (!file) {
      alert('Please select a file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);

    try {
      const response = await fetch('http://localhost:5000/braille', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        setDecodedText(data.text);
        setPreviewSrc(`data:image/jpeg;base64,${data.img_base64}`);
      } else {
        alert(data.message || 'Failed to process image.');
      }

    } catch (err) {
      console.error(err);
      alert('Error uploading image.');
    } finally {
      setLoading(false);
    }
  };
```
