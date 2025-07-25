Imports System.Net
Imports System.Text
Imports System.IO
Imports Newtonsoft.json

Module Program
    Function JsonEscape(value As String) As String
        If value Is Nothing Then Return """"""
        Dim sb As New StringBuilder()
        sb.Append(""""c)
        For Each c As Char In value
            Select Case c
                Case """"c : sb.Append("\""")
                Case "\"c : sb.Append("\\")
                Case "/"c : sb.Append("\/")
                Case ControlChars.Back : sb.Append("\b")
                Case ControlChars.FormFeed : sb.Append("\f")
                Case ControlChars.Lf : sb.Append("\n")
                Case ControlChars.Cr : sb.Append("\r")
                Case ControlChars.Tab : sb.Append("\t")
                Case Else
                    If AscW(c) < 32 Then
                        sb.AppendFormat("\u{0:x4}", AscW(c))
                    Else
                        sb.Append(c)
                    End If
            End Select
        Next
        sb.Append(""""c)
        Return sb.ToString()
    End Function
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
                        formValues(kv(0)) = Uri.UnescapeDataString(kv(1)).Replace("+", " ")
                    End If
                Next

                ' Construire le JSON
                Dim jsonBlocks As New List(Of String)
                For i As Integer = 1 To 3
                    Dim lexicaleValue As String = "false"
                    If formValues.ContainsKey("lexicale") Then
                        lexicaleValue = formValues("lexicale")
                    End If
                    Dim jsonBlock As String = "{" &
                        """filename"":" & JsonEscape(formValues("filename")) & "," &
                        """posteid"":" & JsonEscape(formValues("posteid")) & "," &
                        """userid"":" & JsonEscape(formValues("userid")) & "," &
                        """fonctionPoste"":" & JsonEscape(formValues("fonctionPoste")) & "," &
                        """lexicale"":" & JsonEscape(lexicaleValue) &
                    "}"
                    jsonBlocks.Add(jsonBlock)
                Next
                Dim json As String = "[" & String.Join(",", jsonBlocks) & "]"
                Console.Write(json)
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


