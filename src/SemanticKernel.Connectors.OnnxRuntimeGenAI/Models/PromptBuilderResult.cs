using System;
using System.IO;
using System.Linq;

namespace philabs.SemanticKernel.Connectors.OnnxRuntimeGenAI.Models
{
    internal class PromptBuilderResult
    {
        string _imagePath = "";

        public string Prompt { get; set; }
        public bool ImageFound { get; set; }
        public string? ImagePath
        {
            get
            {

                if (Uri != null && Uri.IsFile)
                {
                    _imagePath = Uri.LocalPath;
                }
                else if (Uri != null && Uri.IsWellFormedUriString(Uri.AbsoluteUri, UriKind.Absolute))
                {
                    // Check if the URI scheme is HTTP or HTTPS
                    if (Uri.Scheme == Uri.UriSchemeHttp || Uri.Scheme == Uri.UriSchemeHttps)
                    {
                        // Additional validation for image file extensions
                        string[] imageExtensions = { ".jpg", ".jpeg", ".png", ".gif" };
                        string fileExtension = Path.GetExtension(Uri.AbsoluteUri);

                        if (imageExtensions.Any(ext => fileExtension.StartsWith(ext)))
                        {
                            string tempFilePath = Path.GetTempFileName();
                            using (var client = new System.Net.WebClient())
                            {
                                client.DownloadFile(Uri.AbsoluteUri, tempFilePath);
                            }
                            _imagePath = tempFilePath;
                        }
                    }
                }
                else if (string.IsNullOrEmpty(_imagePath) && ImageBytes != null && ImageBytes.Length > 0)
                {
                    _imagePath = Path.GetTempFileName();
                    if (File.Exists(_imagePath))
                    {
                        File.Delete(_imagePath);
                    }
                    File.WriteAllBytes(_imagePath, ImageBytes ?? new byte[0]);
                }
                return _imagePath;
            }
            set
            {
                _imagePath = value;
            }
        }
        public byte[]? ImageBytes { get; internal set; }
        public Uri? Uri { get; internal set; }
    }
}
