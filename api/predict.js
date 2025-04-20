export default function handler(req, res) {
  if (req.method === 'POST') {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'No text provided' });
    }

    // تصنيف تجريبي فقط
    let result = 'Halal';
    if (text.toLowerCase().includes('gelatin')) {
      result = 'Haram';
    }

    return res.status(200).json({ result });
  } else {
    return res.status(405).json({ error: 'Method not allowed' });
  }
}
