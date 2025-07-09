Imports System.Net
Imports System.Text
Imports System.IO

Module Program
    Sub Main()
        Dim listener As New HttpListener()
        listener.Prefixes.Add("http://localhost:8080/")
        listener.Start()
        Console.WriteLine("Listening at http://localhost:8080/")

        While True
            Dim context = listener.GetContext()
            Dim request = context.Request
            Dim response = context.Response

            If request.HttpMethod = "POST" AndAlso request.Url.AbsolutePath = "/submit" Then
                ' Read the form data
                Dim body As String
                Using reader As New StreamReader(request.InputStream, request.ContentEncoding)
                    body = reader.ReadToEnd()
                End Using

                ' Parse the form data (simple parsing for "filename" field)
                Dim filenameValue As String = ""
                For Each pair In body.Split("&"c)
                    Dim kv = pair.Split("="c)
                    If kv.Length = 2 AndAlso kv(0) = "filename" Then
                        filenameValue = Uri.UnescapeDataString(kv(1))
                    End If
                Next

                ' ✅ Send to FastAPI
                Dim apiResult As String = ""
                Try
                    Dim fastapiUrl As String = "http://localhost:8000/submit"
                    Dim json As String = "{""param"":""" & filenameValue & """}"
                    Dim data As Byte() = Encoding.UTF8.GetBytes(json)

                    Dim apiRequest As HttpWebRequest = CType(WebRequest.Create(fastapiUrl), HttpWebRequest)
                    apiRequest.Method = "POST"
                    apiRequest.ContentType = "application/json"
                    apiRequest.ContentLength = data.Length

                    Using stream = apiRequest.GetRequestStream()
                        stream.Write(data, 0, data.Length)
                    End Using

                    Using apiResponse = apiRequest.GetResponse()
                        Using apiReader = New StreamReader(apiResponse.GetResponseStream())
                            apiResult = apiReader.ReadToEnd()
                            Console.WriteLine("FastAPI Response: " & apiResult)
                        End Using
                    End Using
                Catch ex As Exception
                    apiResult = "Error calling FastAPI: " & ex.Message
                    Console.WriteLine(apiResult)
                End Try

                ' Show confirmation in the browser
                Dim html As String =
                    "<!DOCTYPE html>" &
                    "<html><body>" &
                    "<h2>File filename: " & filenameValue & " sent to FastAPI.</h2>" &
                    "<p>FastAPI Response: " & apiResult & "</p>" &
                    "<a href='/'>Back</a>" &
                    "</body></html>"

                Dim buffer = Encoding.UTF8.GetBytes(html)
                response.ContentLength64 = buffer.Length
                response.OutputStream.Write(buffer, 0, buffer.Length)
                response.OutputStream.Close()

            Else
                ' Show the form
                Dim html As String =
                    "<!DOCTYPE html>" &
                    "<html><head><meta charset=""UTF-8""></head><body>" &
                    "<form action='/submit' method='post'>" &
                    "  <label for='filename'>filename:</label>" &
                    "  <input type='text' id='filename' name='filename'><br><br>" &
                    "  <input type='submit' value='Submit'>" &
                    "</form>" &
                    "</body></html>"

                Dim buffer = Encoding.UTF8.GetBytes(html)
                response.ContentLength64 = buffer.Length
                response.OutputStream.Write(buffer, 0, buffer.Length)
                response.OutputStream.Close()
            End If
        End While
    End Sub
End Module
