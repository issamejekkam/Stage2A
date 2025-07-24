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
                ' Lire le contenu du formulaire
                Dim body As String
                Using reader As New StreamReader(request.InputStream, request.ContentEncoding)
                    body = reader.ReadToEnd()
                End Using

                ' Extraire toutes les données du formulaire
                Dim formValues As New Dictionary(Of String, String)
                For Each pair In body.Split("&"c)
                    Dim kv = pair.Split("="c)
                    If kv.Length = 2 Then
                        formValues(kv(0)) = Uri.UnescapeDataString(kv(1))
                    End If
                Next

                ' Construire le JSON
                Dim json As String = "{" &
                    """filename"":""" & formValues("filename") & """," &
                    """posteid"":""" & formValues("posteid") & """," &
                    """userid"":""" & formValues("userid") & """," &
                    """fonctionPoste"":""" & formValues("fonctionPoste") & """," &
                    """lexicale"":""" & If(formValues.ContainsKey("lexicale"), formValues("lexicale"), "false") & """" &
                "}"

                ' Envoyer à FastAPI
                Dim apiResult As String = ""
                Try
                    Dim fastapiUrl As String = "http://localhost:8000/submit"
                    Dim data As Byte() = Encoding.UTF8.GetBytes(json)

                    Dim apiRequest As HttpWebRequest = CType(WebRequest.Create(fastapiUrl), HttpWebRequest)
                    apiRequest.Method = "POST"
                    apiRequest.ContentType = "application/json"
                    apiRequest.ContentLength = data.Length
                    apiRequest.Timeout = 600000 
                    apiRequest.ReadWriteTimeout = 600000

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

                ' Réponse HTML
                Dim html As String =
                    "<!DOCTYPE html>" &
                    "<html><body>" &
                    "<h2>Données envoyées à FastAPI.</h2>" &
                    "<p>Réponse: " & apiResult & "</p>" &
                    "<a href='/'>Retour</a>" &
                    "</body></html>"

                Dim buffer = Encoding.UTF8.GetBytes(html)
                response.ContentLength64 = buffer.Length
                response.OutputStream.Write(buffer, 0, buffer.Length)
                response.OutputStream.Close()

            Else
                ' Formulaire HTML
            Dim html As String =
                "<!DOCTYPE html>" &
                "<html><head><meta charset=""UTF-8""></head><body>" &
                "<form action='/submit' method='post'>" &
                "<label for='filename'>filename:</label>" &
                "<input type='text' id='filename' name='filename'><br><br>" &
                "<label for='posteid'>posteid:</label>" &
                "<input type='text' id='posteid' name='posteid'><br><br>" &
                "<label for='userid'>userid:</label>" &
                "<input type='text' id='userid' name='userid'><br><br>" &
                "<label for='fonctionPoste'>fonction/poste ?</label>" &
                "<select id='fonctionPoste' name='fonctionPoste'>" &
                "<option value='FCT'>FCT</option>" &
                "<option value='POS'>POS</option>" &
                "<option value='COL'>COL</option>" &
                "<option value='POSREQ'>POSREQ</option>" &
                "<option value='FCTREQ'>FCTREQ</option>" &
                "</select><br><br>" &
                "<label for='lexicale'>Lexicale ?</label>" &
                "<input type='checkbox' id='lexicale' name='lexicale' value='true'><br><br>" &
                "<input type='submit' value='Submit'>" &
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


